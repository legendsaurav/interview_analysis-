import time
import importlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None

def _load_webrtcvad_module():
    """Load WebRTC VAD module across package naming variants."""
    candidates = [
        "webrtcvad",  # common import path
        "webrtcvad_wheels",  # wheels package may expose Vad here
        "webrtcvad_wheels.webrtcvad",  # alternate nested export
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "Vad"):
                return mod
            nested = getattr(mod, "webrtcvad", None)
            if nested is not None and hasattr(nested, "Vad"):
                return nested
        except Exception:
            continue
    return None


webrtcvad_mod = _load_webrtcvad_module()


@dataclass
class AudioEvent:
    speaking_state: str
    rms: float
    vad_confidence: float
    speaking_duration: float
    transcript: str = ""
    partial_transcript: str = ""
    speech_started: bool = False
    speech_finished: bool = False
    final_pcm16: bytes = b""


class AudioProcessor:
    """Streaming PCM16 audio processor with VAD-like RMS gating and optional Whisper STT."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_seconds: float = 1.0,
        rms_threshold: float = 0.015,
        whisper_model: str = "tiny.en",
        partial_interval_seconds: float = 0.8,
        partial_window_seconds: float = 1.4,
        enable_partial_transcript: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_seconds = silence_seconds
        self.rms_threshold = rms_threshold

        self._in_speech = False
        self._speech_start_ts = 0.0
        self._last_voice_ts = 0.0
        self._total_speaking_duration = 0.0
        self._speech_buffer = bytearray()
        self._partial_interval_seconds = partial_interval_seconds
        self._partial_window_seconds = partial_window_seconds
        self._enable_partial_transcript = enable_partial_transcript
        self._last_partial_emit_ts = 0.0
        self._last_partial_text = ""
        self._noise_floor = 0.003
        self._voice_start_streak = 0
        self._non_voice_accum_seconds = 0.0
        self._vad = None
        if webrtcvad_mod is not None:
            try:
                self._vad = webrtcvad_mod.Vad(2)
            except Exception:
                self._vad = None

        self._stt_model = None
        self._stt_init_error = ""
        self._stt_error_count = 0
        if WhisperModel is not None:
            try:
                # int8 on CPU keeps latency and memory manageable.
                self._stt_model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
            except Exception as exc:
                self._stt_init_error = str(exc)
                print(f"[AudioProcessor] Whisper init failed: {self._stt_init_error}", flush=True)
                self._stt_model = None

    @property
    def has_transcriber(self) -> bool:
        return self._stt_model is not None

    @property
    def has_vad(self) -> bool:
        return self._vad is not None

    def process_chunk(self, pcm16_bytes: bytes, now_ts: Optional[float] = None) -> AudioEvent:
        now_ts = now_ts or time.time()
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if samples.size == 0:
            return AudioEvent("paused", 0.0, 0.0, self._total_speaking_duration)

        norm = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(norm))) + 1e-12)
        chunk_seconds = float(samples.size / max(1, self.sample_rate))
        vad_confidence = self._vad_confidence(pcm16_bytes)

        if not self._in_speech:
            # Track background noise floor only while out of speech.
            self._noise_floor = (self._noise_floor * 0.98) + (rms * 0.02)

        # Adaptive hysteresis thresholds to tolerate ambient room noise.
        start_rms_threshold = max(self.rms_threshold, self._noise_floor * 3.2)
        continue_rms_threshold = max(self.rms_threshold * 0.85, self._noise_floor * 2.4)
        vad_gate = vad_confidence >= 0.6

        if self._in_speech:
            is_voice = vad_gate or rms >= continue_rms_threshold
        else:
            voice_candidate = vad_gate or rms >= start_rms_threshold
            if voice_candidate:
                self._voice_start_streak += 1
            else:
                self._voice_start_streak = 0
            is_voice = self._voice_start_streak >= 2

        speech_started = False
        speech_finished = False
        transcript = ""
        partial_transcript = ""
        final_pcm16 = b""

        if is_voice:
            if not self._in_speech:
                self._in_speech = True
                self._speech_start_ts = now_ts
                self._speech_buffer = bytearray()
                self._last_partial_emit_ts = now_ts
                self._last_partial_text = ""
                self._non_voice_accum_seconds = 0.0
                speech_started = True
            self._last_voice_ts = now_ts
            self._voice_start_streak = 0
            self._non_voice_accum_seconds = 0.0
            self._speech_buffer.extend(pcm16_bytes)

            # Emit partial transcript on a cadence so UI updates within ~1s.
            if (
                self._enable_partial_transcript
                and (now_ts - self._last_partial_emit_ts) >= self._partial_interval_seconds
                and len(self._speech_buffer) >= int(self.sample_rate * 2 * 0.7)
            ):
                partial_transcript = self._transcribe_recent(bytes(self._speech_buffer))
                self._last_partial_emit_ts = now_ts
                if partial_transcript and partial_transcript != self._last_partial_text:
                    self._last_partial_text = partial_transcript
                else:
                    partial_transcript = ""
            speaking_state = "speaking"
        else:
            if self._in_speech:
                self._non_voice_accum_seconds += chunk_seconds

            if self._in_speech and self._non_voice_accum_seconds >= self.silence_seconds:
                chunk_duration = max(0.0, self._last_voice_ts - self._speech_start_ts)
                self._total_speaking_duration += chunk_duration
                self._in_speech = False
                speech_finished = True
                speaking_state = "finished"
                final_pcm16 = bytes(self._speech_buffer)
                self._speech_buffer = bytearray()
                self._last_partial_text = ""
                self._non_voice_accum_seconds = 0.0
            elif self._in_speech:
                speaking_state = "paused"
            else:
                speaking_state = "paused"

        current_duration = self._total_speaking_duration
        if self._in_speech:
            current_duration += max(0.0, now_ts - self._speech_start_ts)

        return AudioEvent(
            speaking_state=speaking_state,
            rms=rms,
            vad_confidence=vad_confidence,
            speaking_duration=current_duration,
            transcript=transcript,
            partial_transcript=partial_transcript,
            speech_started=speech_started,
            speech_finished=speech_finished,
            final_pcm16=final_pcm16,
        )

    def transcribe_bytes(self, pcm16_bytes: bytes) -> str:
        return self._transcribe_bytes(pcm16_bytes)

    def flush_pending(self) -> Dict[str, float | str]:
        """Force-close an active utterance and emit a final transcript."""
        transcript = ""
        if self._in_speech:
            now_ts = time.time()
            self._total_speaking_duration += max(0.0, now_ts - self._speech_start_ts)
            transcript = self._transcribe_bytes(bytes(self._speech_buffer))
            self._in_speech = False
            self._speech_buffer = bytearray()
            self._last_partial_text = ""
        return {
            "speaking_duration": self._total_speaking_duration,
            "transcript": transcript,
        }

    def _transcribe_bytes(self, pcm16_bytes: bytes) -> str:
        if not pcm16_bytes or self._stt_model is None:
            return ""

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segments, _ = self._stt_model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                word_timestamps=False,
                beam_size=1,
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            return text.strip()
        except Exception as exc:
            self._stt_error_count += 1
            if self._stt_error_count <= 3 or (self._stt_error_count % 25 == 0):
                print(f"[AudioProcessor] Whisper transcribe error: {exc}", flush=True)
            return ""

    def _transcribe_recent(self, pcm16_bytes: bytes) -> str:
        if not pcm16_bytes or self._stt_model is None:
            return ""

        max_samples = int(self.sample_rate * self._partial_window_seconds)
        all_samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if all_samples.size > max_samples:
            all_samples = all_samples[-max_samples:]

        audio = all_samples.astype(np.float32) / 32768.0
        try:
            segments, _ = self._stt_model.transcribe(
                audio,
                language="en",
                vad_filter=False,
                word_timestamps=False,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            return text.strip()
        except Exception as exc:
            self._stt_error_count += 1
            if self._stt_error_count <= 3 or (self._stt_error_count % 25 == 0):
                print(f"[AudioProcessor] Whisper partial transcribe error: {exc}", flush=True)
            return ""

    def _vad_confidence(self, pcm16_bytes: bytes) -> float:
        if self._vad is None:
            return 0.0

        # WebRTC VAD supports only 10/20/30 ms frames.
        frame_ms = 30
        frame_samples = int(self.sample_rate * frame_ms / 1000)
        frame_bytes = frame_samples * 2
        if len(pcm16_bytes) < frame_bytes:
            return 0.0

        voiced = 0
        total = 0
        for i in range(0, len(pcm16_bytes) - frame_bytes + 1, frame_bytes):
            frame = pcm16_bytes[i : i + frame_bytes]
            total += 1
            try:
                if self._vad.is_speech(frame, self.sample_rate):
                    voiced += 1
            except Exception:
                return 0.0

        if total == 0:
            return 0.0
        return voiced / total
