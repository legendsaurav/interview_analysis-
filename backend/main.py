import subprocess
from fastapi import APIRouter

router = APIRouter()

# --- Git Pull Endpoint for Lightning AI SadTalker Model ---
@router.post("/api/git/pull")
async def git_pull():
    """
    Pulls the latest changes from the remote GitHub repository.
    """
    try:
        result = subprocess.run(["git", "pull"], cwd=str(BASE_DIR.parent), capture_output=True, text=True, timeout=20)
        return {"ok": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --- Git Status Endpoint ---
@router.get("/api/git/status")
async def git_status():
    try:
        result = subprocess.run(["git", "status", "-sb"], cwd=str(BASE_DIR.parent), capture_output=True, text=True, timeout=10)
        return {"ok": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Register router
app.include_router(router)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/legendsaurav/changer/main/live_transcript_latest.txt"

async def fetch_transcript_from_github() -> str:
    """
    Fetches the transcript text file from the GitHub repo.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(GITHUB_RAW_URL)
        resp.raise_for_status()
        return resp.text

async def load_transcript_lines_from_github() -> list[str]:
    text = await fetch_transcript_from_github()
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "[FINAL]" in line:
            payload = re.sub(r"^\[[^\]]+\]\s+\[[^\]]+\]\s+\[[^\]]+\]\s*", "", line).strip()
            if payload:
                lines.append(payload)
        elif "[PARTIAL]" not in line and len(line.split()) >= 4:
            lines.append(line)
    return lines
from __future__ import annotations

import asyncio
import base64
from collections import deque
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
import uuid

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import httpx
# --- Bridge Model ---
from pydantic import BaseModel
class InterviewerSelection(BaseModel):
    interviewer_id: str
    interviewer_name: str
    user_id: str | None = None
    extra: dict = {}

# --- Bridge Endpoint ---
app = FastAPI(title="Realtime Interview System")

@app.post("/api/interview/select")
async def select_interviewer(payload: InterviewerSelection, request: Request):
    """
    Receives interviewer selection and forwards to INTERVIEWER and lost backends.
    """
    # Prevent infinite loop: only forward if not already from another backend
    source = request.headers.get("X-Bridge-Source", "yolo_yolov5")
    results = {}
    if source != "INTERVIEWER":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post("http://localhost:8000/api/interview/select", json=payload.dict(), headers={"X-Bridge-Source": "yolo_yolov5"})
                results["INTERVIEWER"] = resp.json()
        except Exception as e:
            results["INTERVIEWER"] = {"error": str(e)}
    if source != "lost":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post("http://localhost:8787/api/interview/select", json=payload.dict(), headers={"X-Bridge-Source": "yolo_yolov5"})
                results["lost"] = resp.json()
        except Exception as e:
            results["lost"] = {"error": str(e)}
    return {"ok": True, "forwarded": results, "received": payload.dict()}
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio_processor import AudioProcessor
from .gemini_service import GeminiInterviewAssistant
from .state_machine import InterviewState, InterviewStateMachine
from .vision_processor import VisionProcessor

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"


def _resolve_yolo_model_path() -> Path:
    override = os.getenv("INTERVIEW_YOLO_PATH", "").strip()
    if override:
        p = Path(override)
        if p.exists():
            return p

    candidates = [
        BASE_DIR.parent.parent.parent / "yolov8n.pt",  # repo root
        BASE_DIR.parent.parent / "yolov8n.pt",  # legacy layout fallback
        BASE_DIR.parent / "yolov8n.pt",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Return the expected repo-root path even if missing so logs/config are deterministic.
    return BASE_DIR.parent.parent.parent / "yolov8n.pt"


YOLO_MODEL_PATH = _resolve_yolo_model_path()
SESSION_TRANSCRIPTS_DIR = BASE_DIR / "session_transcripts"
LEGACY_TRANSCRIPT_LATEST_PATH = BASE_DIR / "live_transcript_latest.txt"
LEGACY_TRANSCRIPT_LOG_PATH = BASE_DIR / "live_transcript_log.txt"
CLIPS_DIR = BASE_DIR / "clips"
CHAT_TTL_SECONDS = int(os.getenv("INTERVIEW_CHAT_TTL_SECONDS", str(5 * 60)))
TRANSCRIPT_RETENTION_SECONDS = int(os.getenv("INTERVIEW_TRANSCRIPT_TTL_SECONDS", str(CHAT_TTL_SECONDS)))

DEBUG_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=250)
SESSION_TRANSCRIPT_FILES: Dict[int, Dict[str, Any]] = {}
SESSION_CLIP_DIRS: Dict[int, Dict[str, Any]] = {}
CLEANUP_TASK: asyncio.Task | None = None
SESSION_ANALYTICS: Dict[int, Dict[str, Any]] = {}
GEMINI_ASSISTANT = GeminiInterviewAssistant(
    api_key=os.getenv("GEMINI_API_KEY", ""),
    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
)
DEBUG_RUNTIME: Dict[str, Any] = {
    "connected_clients": 0,
    "last_update_ts": 0.0,
    "last_audio_chunk_bytes": 0,
    "last_video_frame_bytes": 0,
    "last_audio_state": "paused",
    "last_vad_confidence": 0.0,
    "last_partial_transcript": "",
    "last_final_transcript": "",
    "last_vision": {},
    "last_persisted_text": "",
    "latest_transcript_file": "",
    "transcript_log_file": "",
    "last_transcription_latency_ms": 0.0,
    "gemini_configured": GEMINI_ASSISTANT.configured,
    "last_final_report": {},
}


class GeminiConfigRequest(BaseModel):
    api_key: str


class AnalyzeTranscriptRequest(BaseModel):
    transcript_path: str
    session_id: Optional[int] = None


def _chat_cutoff_ts(now: float | None = None) -> float:
    current = time.time() if now is None else now
    return current - CHAT_TTL_SECONDS


def _transcript_cutoff_ts(now: float | None = None) -> float:
    current = time.time() if now is None else now
    return current - TRANSCRIPT_RETENTION_SECONDS


def _prune_debug_events(now: float | None = None) -> None:
    cutoff = _chat_cutoff_ts(now)
    while DEBUG_EVENTS and float(DEBUG_EVENTS[0].get("ts", 0.0)) < cutoff:
        DEBUG_EVENTS.popleft()


def _cleanup_expired_session_transcript_files(now: float | None = None) -> None:
    current = time.time() if now is None else now
    cutoff = _transcript_cutoff_ts(current)
    expired_sessions = []

    for session_id, info in SESSION_TRANSCRIPT_FILES.items():
        created_at = float(info.get("created_at", 0.0))
        if created_at < cutoff:
            for key in ("latest_path", "log_path"):
                p = info.get(key)
                if isinstance(p, Path) and p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        SESSION_TRANSCRIPT_FILES.pop(session_id, None)


def _cleanup_transcript_files_on_disk(now: float | None = None) -> None:
    current = time.time() if now is None else now
    cutoff = _transcript_cutoff_ts(current)
    if not SESSION_TRANSCRIPTS_DIR.exists():
        return

    for p in SESSION_TRANSCRIPTS_DIR.glob("live_transcript_*.txt"):
        try:
            if not p.is_file():
                continue
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except Exception:
            pass


def _cleanup_expired_session_clips(now: float | None = None) -> None:
    current = time.time() if now is None else now
    cutoff = _chat_cutoff_ts(current)
    expired_sessions = []

    for session_id, info in SESSION_CLIP_DIRS.items():
        created_at = float(info.get("created_at", 0.0))
        if created_at < cutoff:
            clip_dir = info.get("clip_dir")
            if isinstance(clip_dir, Path) and clip_dir.exists() and clip_dir.is_dir():
                for p in clip_dir.glob("*"):
                    try:
                        if p.is_file():
                            p.unlink()
                    except Exception:
                        pass
                try:
                    clip_dir.rmdir()
                except Exception:
                    pass
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        SESSION_CLIP_DIRS.pop(session_id, None)


def _cleanup_clip_dirs_on_disk(now: float | None = None) -> None:
    current = time.time() if now is None else now
    cutoff = _chat_cutoff_ts(current)
    if not CLIPS_DIR.exists():
        return

    for p in CLIPS_DIR.iterdir():
        try:
            if not p.is_dir():
                continue
            if p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


async def _cleanup_loop() -> None:
    while True:
        now = time.time()
        _prune_debug_events(now)
        _cleanup_expired_session_transcript_files(now)
        _cleanup_transcript_files_on_disk(now)
        _cleanup_expired_session_clips(now)
        _cleanup_clip_dirs_on_disk(now)
        await asyncio.sleep(15)


def _debug_record(event_type: str, payload: Dict[str, Any]) -> None:
    _prune_debug_events()
    ts = round(time.time(), 3)
    DEBUG_RUNTIME["last_update_ts"] = ts
    DEBUG_EVENTS.append({"ts": ts, "type": event_type, **payload})


def _console_transcript(source: str, text: str, is_final: bool = False) -> None:
    content = text.strip()
    if not content:
        return
    clipped = content if len(content) <= 240 else content[:237] + "..."
    stamp = time.strftime("%H:%M:%S")
    final_tag = "FINAL" if is_final else "PARTIAL"
    print(f"[{stamp}] [TRANSCRIPT:{final_tag}] [{source}] {clipped}", flush=True)


def _persist_transcript(runtime: "SessionRuntime", source: str, text: str, is_final: bool = False) -> None:
    content = text.strip()
    if not content:
        return

    if runtime.transcript_files_expired:
        return

    now = time.time()
    if now - runtime.transcript_created_at > TRANSCRIPT_RETENTION_SECONDS:
        runtime.transcript_files_expired = True
        for p in (runtime.transcript_latest_path, runtime.transcript_log_path):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        SESSION_TRANSCRIPT_FILES.pop(runtime.session_id, None)
        return

    last = ""
    if runtime.transcript_history:
        last = str(runtime.transcript_history[-1].get("text", "")).strip()

    # Avoid writing duplicate sync payloads repeatedly.
    if content == last:
        return

    # Suppress unstable partial regressions; keep latest stronger sentence.
    if not is_final and last:
        # If new partial is significantly shorter or fully contained in last text,
        # it is likely a recognizer rollback jitter.
        if len(content) + 12 < len(last):
            return
        if content in last:
            return

    # Normalize spacing-only changes.
    if " ".join(content.split()) == " ".join(last.split()):
        return

    runtime.transcript_history.append(
        {
            "ts": now,
            "source": source,
            "final": bool(is_final),
            "text": content,
        }
    )

    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    final_tag = "FINAL" if is_final else "PARTIAL"
    runtime.transcript_latest_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.transcript_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime.transcript_latest_path.write_text(content + "\n", encoding="utf-8")
        with runtime.transcript_log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] [{final_tag}] [{source}] {content}\n")
    except Exception as exc:
        _debug_record(
            "transcript_write_error",
            {
                "session_id": runtime.session_id,
                "source": source,
                "error": str(exc),
            },
        )
        return

    DEBUG_RUNTIME["last_persisted_text"] = content


def _persist_transcript_by_session_id(session_id: int, source: str, text: str, is_final: bool = False) -> bool:
    info = SESSION_TRANSCRIPT_FILES.get(session_id)
    if not info:
        return False

    now = time.time()
    created_at = float(info.get("created_at", 0.0))
    if now - created_at > TRANSCRIPT_RETENTION_SECONDS:
        _cleanup_expired_session_transcript_files(now)
        return False

    content = text.strip()
    if not content:
        return False

    latest_path = info.get("latest_path")
    log_path = info.get("log_path")
    if not isinstance(latest_path, Path) or not isinstance(log_path, Path):
        return False

    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    final_tag = "FINAL" if is_final else "PARTIAL"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        latest_path.write_text(content + "\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] [{final_tag}] [{source}] {content}\n")
    except Exception:
        return False
    return True


def _record_timeline_event(runtime: "SessionRuntime", event: str, confidence: float, details: Optional[Dict[str, Any]] = None) -> None:
    ts = round(time.time() - runtime.started_at, 2)
    runtime.timeline_events.append(
        {
            "timestamp": ts,
            "event": event,
            "confidence": round(float(confidence), 3),
            "details": details or {},
        }
    )
    runtime.timeline_version += 1


def _calc_scores(runtime: "SessionRuntime") -> Dict[str, float]:
    transcript = runtime.last_transcript or ""
    words = len(transcript.split())
    speech = max(0.0, runtime.latest_speaking_duration)

    relevance_score = min(100.0, (words / 70.0) * 100.0)
    completeness_score = min(100.0, (speech / 25.0) * 100.0)
    answer_quality = round((0.55 * relevance_score) + (0.45 * completeness_score), 2)

    hesitation_penalty = min(45.0, runtime.pause_count * 4.0)
    continuity_score = max(0.0, 100.0 - hesitation_penalty)
    confidence_score = round((0.7 * continuity_score) + (0.3 * (runtime.avg_eye_contact * 100.0)), 2)

    cheating_penalty = (
        runtime.event_counts.get("phone_detected", 0) * 22
        + runtime.event_counts.get("multiple_persons", 0) * 25
        + runtime.event_counts.get("multimodal_suspicion", 0) * 12
        + runtime.event_counts.get("suspicious_movement", 0) * 8
    )
    cheating_score = round(min(100.0, float(cheating_penalty)), 2)

    final_score = round((0.5 * answer_quality) + (0.3 * confidence_score) - (0.2 * cheating_score), 2)
    return {
        "answer_quality": answer_quality,
        "confidence": confidence_score,
        "cheating_risk": cheating_score,
        "final_score": max(0.0, min(100.0, final_score)),
    }


def _build_final_report(runtime: "SessionRuntime") -> Dict[str, Any]:
    scores = _calc_scores(runtime)
    timeline = list(runtime.timeline_events)[-80:]
    return {
        "pipeline": [
            "User joins interview",
            "Baseline environment captured",
            "Real-time audio + video processing",
            "Events + clips stored",
            "Post-interview AI analysis (<=5 min)",
            "Final evaluation report generated",
        ],
        "audio_signals": {
            "speech_duration": round(runtime.latest_speaking_duration, 2),
            "pause_count": runtime.pause_count,
            "long_silence": runtime.long_silence,
        },
        "vision": {
            "baseline_objects": runtime.latest_vision.get("baseline_objects", []),
            "conf_threshold": 0.5,
            "imgsz": 320,
            "cpu_mode": True,
        },
        "scores": scores,
        "timeline": timeline,
        "clips": runtime.saved_clips,
        "ai_answer_analysis": runtime.answer_evaluations,
        "ai_accuracy_series": runtime.ai_accuracy_series,
        "clip_analysis_framework": {
            "recommended_model": "YOLOv8m or YOLOv8l",
            "recommended_imgsz": 640,
            "purpose": "verify and refine suspicious detections from realtime stage",
        },
    }


def _append_frame_for_clip(runtime: "SessionRuntime", now: float, jpeg_bytes: bytes) -> None:
    runtime.frame_buffer.append({"ts": now, "jpeg": jpeg_bytes})
    cutoff = now - 2.2
    while runtime.frame_buffer and float(runtime.frame_buffer[0]["ts"]) < cutoff:
        runtime.frame_buffer.popleft()


def _trigger_clip(runtime: "SessionRuntime", event_name: str, confidence: float, now: float) -> None:
    clip_id = f"{runtime.session_id}_{int(now * 1000)}_{event_name}"
    pre = [item for item in runtime.frame_buffer if float(item["ts"]) >= now - 2.0]
    runtime.pending_clips.append(
        {
            "clip_id": clip_id,
            "event": event_name,
            "confidence": round(confidence, 3),
            "start_ts": now - 2.0,
            "end_ts": now + 2.0,
            "frames": pre,
            "saved": False,
        }
    )


def _flush_completed_clips(runtime: "SessionRuntime", now: float) -> None:
    for clip in runtime.pending_clips:
        if clip["saved"]:
            continue
        if now <= float(clip["end_ts"]):
            continue

        frames: List[Dict[str, Any]] = clip["frames"]
        if len(frames) < 4:
            clip["saved"] = True
            continue

        runtime.clips_dir.mkdir(parents=True, exist_ok=True)
        clip_path = runtime.clips_dir / f"{clip['clip_id']}.mp4"
        first = cv2.imdecode(np.frombuffer(frames[0]["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            clip["saved"] = True
            continue

        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
        for item in frames:
            img = cv2.imdecode(np.frombuffer(item["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
        writer.release()

        runtime.saved_clips.append(
            {
                "clip_id": clip["clip_id"],
                "event": clip["event"],
                "confidence": clip["confidence"],
                "path": str(clip_path),
            }
        )
        clip["saved"] = True

    runtime.pending_clips = [c for c in runtime.pending_clips if not c.get("saved")]


def _collect_question_history(runtime: "SessionRuntime") -> List[str]:
    out: List[str] = []
    for item in runtime.answer_evaluations:
        q = str(item.get("question", "")).strip()
        if q:
            out.append(q)
    active_q = str(runtime.active_question_bundle.get("question", "")).strip()
    if active_q:
        out.append(active_q)
    return out[-10:]


async def _next_question_bundle(runtime: "SessionRuntime", requested_question: Optional[str] = None) -> Dict[str, Any]:
    direct = (requested_question or "").strip()
    if direct:
        ideal = "Use a structured answer: context, actions you took, and measurable result."
        return {
            "question": direct,
            "ideal_answer": ideal,
            "key_points": ["context", "actions", "result"],
            "source": "manual",
        }

    if GEMINI_ASSISTANT.configured:
        try:
            prior = _collect_question_history(runtime)
            bundle = await asyncio.to_thread(GEMINI_ASSISTANT.generate_question_answer, prior)
            question = str(bundle.get("question", "")).strip()
            ideal = str(bundle.get("ideal_answer", "")).strip()
            key_points = bundle.get("key_points", [])
            if question and ideal:
                return {
                    "question": question,
                    "ideal_answer": ideal,
                    "key_points": [str(x).strip() for x in key_points if str(x).strip()][:6],
                    "source": str(bundle.get("source", "gemini")),
                }
        except Exception as exc:
            _debug_record("gemini_question_error", {"error": str(exc)})

    if runtime.state.question_bank:
        next_idx = (runtime.fallback_question_index + 1) % len(runtime.state.question_bank)
        runtime.fallback_question_index = next_idx
        question = str(runtime.state.question_bank[next_idx]).strip()
    else:
        question = "Tell me about yourself."
    return {
        "question": question,
        "ideal_answer": "Answer clearly with context, your action, and a measurable outcome.",
        "key_points": ["context", "action", "outcome"],
        "source": "fallback",
    }


def _extract_answer_text_since(runtime: "SessionRuntime", since_ts: float) -> str:
    parts: List[str] = []
    seen: set[str] = set()
    for item in runtime.transcript_history:
        try:
            ts = float(item.get("ts", 0.0))
        except Exception:
            ts = 0.0
        if ts < since_ts:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        parts.append(text)

    if not parts:
        return str(runtime.last_transcript or "").strip()
    return " ".join(parts).strip()


async def _evaluate_current_answer(runtime: "SessionRuntime") -> Optional[Dict[str, Any]]:
    bundle = runtime.active_question_bundle
    question = str(bundle.get("question", "")).strip()
    ideal_answer = str(bundle.get("ideal_answer", "")).strip()
    if not question:
        return None

    answer_text = _extract_answer_text_since(runtime, float(runtime.question_started_at))
    if not answer_text:
        return None

    try:
        evaluation = await asyncio.to_thread(
            GEMINI_ASSISTANT.evaluate_answer,
            question,
            ideal_answer,
            answer_text,
        )
    except Exception as exc:
        _debug_record("gemini_eval_error", {"error": str(exc)})
        evaluation = {
            "score": 0.0,
            "strengths": [],
            "gaps": ["Evaluation failed."],
            "feedback": "Could not evaluate this answer due to an AI service error.",
            "source": "error",
        }

    score = max(0.0, min(100.0, float(evaluation.get("score", 0.0))))
    item = {
        "index": len(runtime.answer_evaluations) + 1,
        "question": question,
        "ideal_answer": ideal_answer,
        "user_answer": answer_text,
        "accuracy": round(score, 2),
        "strengths": list(evaluation.get("strengths", []))[:5],
        "gaps": list(evaluation.get("gaps", []))[:5],
        "feedback": str(evaluation.get("feedback", "")).strip(),
        "source": str(evaluation.get("source", "rule")),
        "ts": time.time(),
    }
    runtime.answer_evaluations.append(item)
    runtime.ai_accuracy_series = [
        {"x": idx + 1, "label": f"Q{idx + 1}", "y": float(v.get("accuracy", 0.0))}
        for idx, v in enumerate(runtime.answer_evaluations)
    ]
    return item


def _load_transcript_lines(path: Path) -> List[str]:
    if not path.exists() or not path.is_file():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "[FINAL]" in line:
            payload = re.sub(r"^\[[^\]]+\]\s+\[[^\]]+\]\s+\[[^\]]+\]\s*", "", line).strip()
            if payload:
                lines.append(payload)
        elif "[PARTIAL]" not in line and len(line.split()) >= 4:
            lines.append(line)
    return lines


async def _run_transcription_queue(runtime: "SessionRuntime", websocket: WebSocket) -> None:
    try:
        while runtime.pending_transcriptions:
            payload = runtime.pending_transcriptions.popleft()
            pcm16 = payload.get("pcm16", b"")
            duration = float(payload.get("duration", 0.0))
            if not isinstance(pcm16, (bytes, bytearray)) or not pcm16:
                continue

            started = time.time()
            text = await asyncio.to_thread(runtime.audio.transcribe_bytes, bytes(pcm16))
            latency_ms = round((time.time() - started) * 1000.0, 1)
            DEBUG_RUNTIME["last_transcription_latency_ms"] = latency_ms

            if not text:
                continue

            runtime.last_transcript = text
            DEBUG_RUNTIME["last_final_transcript"] = text
            _debug_record(
                "audio_final_async",
                {
                    "session_id": runtime.session_id,
                    "duration": round(duration, 2),
                    "latency_ms": latency_ms,
                    "text": text,
                },
            )
            _console_transcript(source="whisper_final_async", text=text, is_final=True)
            _persist_transcript(runtime=runtime, source="whisper_final_async", text=text, is_final=True)

            try:
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "final": True,
                    }
                )
            except Exception:
                break
    finally:
        runtime.transcription_task = None


app = FastAPI(title="Realtime Interview System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_cleanup_task() -> None:
    global CLEANUP_TASK
    SESSION_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    _prune_debug_events()
    _cleanup_expired_session_transcript_files()
    _cleanup_transcript_files_on_disk()
    _cleanup_expired_session_clips()
    _cleanup_clip_dirs_on_disk()
    CLEANUP_TASK = asyncio.create_task(_cleanup_loop())


@app.on_event("shutdown")
async def _shutdown_cleanup_task() -> None:
    global CLEANUP_TASK
    if CLEANUP_TASK:
        CLEANUP_TASK.cancel()
        try:
            await CLEANUP_TASK
        except asyncio.CancelledError:
            pass
        CLEANUP_TASK = None

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/styles.css", include_in_schema=False)
async def frontend_styles() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "styles.css"))


@app.get("/app.js", include_in_schema=False)
async def frontend_app_js() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "app.js"))


@app.get("/runtime-config.js", include_in_schema=False)
async def frontend_runtime_config() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "runtime-config.js"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    # Return empty favicon response to silence browser 404 retries.
    return Response(status_code=204)


@app.get("/debug/ingest")
async def debug_ingest() -> Dict[str, Any]:
    return {
        "runtime": DEBUG_RUNTIME,
        "events": list(DEBUG_EVENTS)[-80:],
    }


@app.get("/debug/final-report")
async def debug_final_report() -> Dict[str, Any]:
    return {
        "report": DEBUG_RUNTIME.get("last_final_report", {}),
    }


@app.get("/ai/gemini/status")
async def gemini_status() -> Dict[str, Any]:
    return {
        "configured": bool(DEBUG_RUNTIME.get("gemini_configured", False)),
        "model": GEMINI_ASSISTANT.model,
        "last_error": GEMINI_ASSISTANT.last_error,
    }


@app.post("/ai/gemini/config")
async def gemini_config(payload: GeminiConfigRequest) -> Dict[str, Any]:
    key_set = GEMINI_ASSISTANT.configure_api_key(payload.api_key)
    verified = False
    if key_set:
        verified = await asyncio.to_thread(GEMINI_ASSISTANT.verify_connection)
        if not verified:
            GEMINI_ASSISTANT.configure_api_key("")

    DEBUG_RUNTIME["gemini_configured"] = bool(verified)
    return {
        "ok": bool(verified),
        "configured": bool(verified),
        "model": GEMINI_ASSISTANT.model,
        "last_error": GEMINI_ASSISTANT.last_error,
    }


@app.post("/analysis/transcript-file")
async def analyze_transcript_file(payload: AnalyzeTranscriptRequest) -> Dict[str, Any]:
    path = Path(payload.transcript_path)
    lines = _load_transcript_lines(path)
    if not lines:
        return {
            "ok": False,
            "reason": "no_lines",
            "evaluations": [],
            "accuracy_series": [],
        }

    session_data = SESSION_ANALYTICS.get(int(payload.session_id or 0), {})
    qa_ref = list(session_data.get("answer_evaluations", []))

    evaluations: List[Dict[str, Any]] = []
    for idx, answer in enumerate(lines):
        if idx < len(qa_ref):
            question = str(qa_ref[idx].get("question", "Tell me about yourself.")).strip()
            ideal = str(qa_ref[idx].get("ideal_answer", "Give a structured answer with measurable impact.")).strip()
        else:
            question = "Tell me about yourself."
            ideal = "Give a concise, role-relevant summary, key strengths, and one quantified achievement."

        try:
            result = await asyncio.to_thread(GEMINI_ASSISTANT.evaluate_answer, question, ideal, answer)
        except Exception as exc:
            result = {
                "score": 0.0,
                "strengths": [],
                "gaps": [f"Evaluation failed: {exc}"],
                "feedback": "Could not analyze this answer.",
                "source": "error",
            }

        score = max(0.0, min(100.0, float(result.get("score", 0.0))))
        evaluations.append(
            {
                "index": idx + 1,
                "question": question,
                "ideal_answer": ideal,
                "user_answer": answer,
                "accuracy": round(score, 2),
                "strengths": list(result.get("strengths", []))[:5],
                "gaps": list(result.get("gaps", []))[:5],
                "feedback": str(result.get("feedback", "")).strip(),
            }
        )

    accuracy_series = [
        {"x": i + 1, "label": f"Q{i + 1}", "y": float(item.get("accuracy", 0.0))}
        for i, item in enumerate(evaluations)
    ]

    return {
        "ok": True,
        "evaluations": evaluations,
        "accuracy_series": accuracy_series,
        "source_file": str(path),
    }


@app.get("/analysis/session/{session_id}")
async def analysis_by_session(session_id: int) -> Dict[str, Any]:
    data = SESSION_ANALYTICS.get(session_id)
    if not data:
        return {
            "ok": False,
            "reason": "session_not_found",
            "evaluations": [],
            "accuracy_series": [],
        }
    return {
        "ok": True,
        "session_id": session_id,
        "evaluations": data.get("answer_evaluations", []),
        "accuracy_series": data.get("accuracy_series", []),
        "report": data.get("report", {}),
    }


@app.post("/debug/ingest/transcript")
async def debug_ingest_transcript(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    is_final = bool(payload.get("final", False))
    source = str(payload.get("source", "frontend_http"))
    session_id_raw = payload.get("session_id")
    if not text:
        return {"ok": False, "reason": "empty_text"}

    if is_final:
        DEBUG_RUNTIME["last_final_transcript"] = text
    else:
        DEBUG_RUNTIME["last_partial_transcript"] = text

    _debug_record(
        "http_transcript_sync",
        {
            "source": source,
            "final": is_final,
            "text": text,
        },
    )
    _console_transcript(source=source, text=text, is_final=is_final)
    if session_id_raw is None:
        return {"ok": False, "reason": "missing_session_id"}
    try:
        session_id = int(session_id_raw)
    except Exception:
        return {"ok": False, "reason": "invalid_session_id"}

    ok = _persist_transcript_by_session_id(session_id=session_id, source=source, text=text, is_final=is_final)
    return {"ok": ok}


class SessionRuntime:
    def __init__(self) -> None:
        self.session_id = int(time.time() * 1000)
        self.started_at = time.time()
        SESSION_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        token = f"{self.session_id}_{uuid.uuid4().hex[:10]}"
        self.transcript_latest_path = SESSION_TRANSCRIPTS_DIR / f"live_transcript_latest_{token}.txt"
        self.transcript_log_path = SESSION_TRANSCRIPTS_DIR / f"live_transcript_log_{token}.txt"
        self.transcript_latest_path.write_text("", encoding="utf-8")
        self.transcript_log_path.write_text("", encoding="utf-8")
        self.transcript_created_at = self.started_at
        self.transcript_files_expired = False
        self.transcript_history: Deque[Dict[str, Any]] = deque(maxlen=500)

        SESSION_TRANSCRIPT_FILES[self.session_id] = {
            "created_at": self.transcript_created_at,
            "latest_path": self.transcript_latest_path,
            "log_path": self.transcript_log_path,
        }

        DEBUG_RUNTIME["latest_transcript_file"] = str(self.transcript_latest_path)
        DEBUG_RUNTIME["transcript_log_file"] = str(self.transcript_log_path)

        # Keep legacy files as pointers so users can quickly find the active session files.
        LEGACY_TRANSCRIPT_LATEST_PATH.write_text(
            "This file is deprecated for shared writes.\n"
            f"Active session latest transcript file:\n{self.transcript_latest_path}\n",
            encoding="utf-8",
        )
        with LEGACY_TRANSCRIPT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"session_id={self.session_id} "
                f"latest={self.transcript_latest_path} "
                f"log={self.transcript_log_path}\n"
            )
        self.audio = AudioProcessor(
            sample_rate=16000,
            silence_seconds=1.0,
            whisper_model="tiny.en",
            enable_partial_transcript=False,
        )
        self.accept_client_transcript = not self.audio.has_transcriber
        self.transcript_mode = "whisper-primary" if self.audio.has_transcriber else "browser-fallback"
        self.vision = VisionProcessor(yolo_model_path=str(YOLO_MODEL_PATH), yolo_every_n_frames=4)
        self.state = InterviewStateMachine(
            expected_answer_seconds=20.0,
            max_answer_seconds=60.0,
            post_speech_pause_seconds=2.5,
        )
        self.latest_vision: Dict[str, object] = {
            "eye_contact_score": 0.5,
            "head_movement_score": 0.5,
            "gaze_direction": "unknown",
            "detected_objects": [],
            "alerts": [],
        }
        self.last_transcript = ""
        self.last_backend_response = ""
        self.model_available = self.vision.detector.is_available
        self.speech_mismatch_frames = 0
        self.audio_chunks = 0
        self.video_frames = 0
        self.prev_speaking_state = "paused"
        self.pause_started_at = 0.0
        self.pause_count = 0
        self.long_silence = False
        self.latest_speaking_duration = 0.0
        self.last_audio_total_speaking_duration = 0.0
        self.question_start_speaking_duration = 0.0
        self.auto_next_cooldown_until = 0.0
        self.question_started_at = self.started_at
        self.timeline_events: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.timeline_version = 0
        self.last_timeline_sent_version = 0
        self.last_timeline_emit_ts = 0.0
        self.event_counts: Dict[str, int] = {
            "phone_detected": 0,
            "multiple_persons": 0,
            "suspicious_movement": 0,
            "look_away": 0,
            "silence": 0,
            "long_pause": 0,
            "multimodal_suspicion": 0,
        }
        self.avg_eye_contact = 0.5
        self.eye_samples = 0
        self.frame_buffer: Deque[Dict[str, Any]] = deque(maxlen=80)
        self.pending_clips: List[Dict[str, Any]] = []
        self.clips_dir = CLIPS_DIR / str(self.session_id)
        SESSION_CLIP_DIRS[self.session_id] = {
            "created_at": self.started_at,
            "clip_dir": self.clips_dir,
        }
        self.saved_clips: List[Dict[str, Any]] = []
        self.last_report: Dict[str, Any] = {}
        self.interview_active = False
        self.pending_transcriptions: Deque[Dict[str, Any]] = deque(maxlen=8)
        self.transcription_task: asyncio.Task | None = None
        self.last_sync_persist_ts = 0.0
        self.last_sync_persist_text = ""
        self.active_question_bundle: Dict[str, Any] = {}
        self.answer_evaluations: List[Dict[str, Any]] = []
        self.ai_accuracy_series: List[Dict[str, Any]] = []
        self.fallback_question_index = -1


@app.websocket("/ws/interview")
async def interview_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    runtime = SessionRuntime()
    DEBUG_RUNTIME["connected_clients"] = int(DEBUG_RUNTIME.get("connected_clients", 0)) + 1
    _debug_record("session_connected", {"session_id": runtime.session_id})

    await websocket.send_json(
        {
            "type": "system",
            "message": "Connected. Press Start Interview to begin.",
            "session_id": runtime.session_id,
            "latest_transcript_file": str(runtime.transcript_latest_path),
            "transcript_log_file": str(runtime.transcript_log_path),
            "transcript_mode": runtime.transcript_mode,
            "state": runtime.state.state.value,
            "stt_available": runtime.audio.has_transcriber,
            "vad_available": runtime.audio.has_vad,
            "model_available": runtime.model_available,
            "gemini_configured": GEMINI_ASSISTANT.configured,
            "yolo_model_path": str(YOLO_MODEL_PATH),
        }
    )

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            if kind == "control":
                action = msg.get("action")
                _debug_record("control", {"session_id": runtime.session_id, "action": action})
                if action == "start":
                    runtime.interview_active = True
                    runtime.started_at = time.time()
                    runtime.question_started_at = runtime.started_at
                    runtime.question_start_speaking_duration = runtime.last_audio_total_speaking_duration
                    runtime.auto_next_cooldown_until = 0.0
                    bundle = await _next_question_bundle(runtime, msg.get("question"))
                    runtime.active_question_bundle = bundle
                    event = runtime.state.ask_question(bundle.get("question"))
                    _record_timeline_event(runtime, "interview_started", 1.0, {"question": event.get("question", "")})
                    await websocket.send_json({"type": "state", **event, "ideal_answer": bundle.get("ideal_answer", "")})
                elif action == "next":
                    eval_item = await _evaluate_current_answer(runtime)
                    if eval_item:
                        await websocket.send_json(
                            {
                                "type": "qa_analysis_update",
                                "latest": eval_item,
                                "series": runtime.ai_accuracy_series,
                            }
                        )
                    runtime.question_started_at = time.time()
                    runtime.question_start_speaking_duration = runtime.last_audio_total_speaking_duration
                    runtime.auto_next_cooldown_until = 0.0
                    bundle = await _next_question_bundle(runtime, msg.get("question"))
                    runtime.active_question_bundle = bundle
                    event = runtime.state.reset_for_next_question(bundle.get("question"))
                    await websocket.send_json({"type": "state", **event, "ideal_answer": bundle.get("ideal_answer", "")})
                elif action == "stop":
                    runtime.interview_active = False
                    if runtime.transcription_task and not runtime.transcription_task.done():
                        runtime.transcription_task.cancel()
                    flushed = runtime.audio.flush_pending()
                    if flushed.get("transcript"):
                        runtime.last_transcript = str(flushed["transcript"])
                    eval_item = await _evaluate_current_answer(runtime)
                    if eval_item:
                        await websocket.send_json(
                            {
                                "type": "qa_analysis_update",
                                "latest": eval_item,
                                "series": runtime.ai_accuracy_series,
                            }
                        )
                    runtime.prev_speaking_state = "paused"
                    runtime.pause_started_at = 0.0
                    runtime.long_silence = False
                    runtime.state.state = InterviewState.IDLE
                    runtime.last_report = _build_final_report(runtime)
                    SESSION_ANALYTICS[runtime.session_id] = {
                        "answer_evaluations": runtime.answer_evaluations,
                        "accuracy_series": runtime.ai_accuracy_series,
                        "report": runtime.last_report,
                        "updated_at": time.time(),
                    }
                    DEBUG_RUNTIME["last_final_report"] = runtime.last_report
                    await websocket.send_json(
                        {
                            "type": "audio_metrics",
                            "speaking_state": "paused",
                            "speaking_duration": 0.0,
                            "rms": 0.0,
                            "vad_confidence": 0.0,
                            "pause_count": runtime.pause_count,
                            "long_silence": False,
                        }
                    )
                    await websocket.send_json({"type": "state", "state": InterviewState.IDLE.value, "question": runtime.state.active_question})
                    await websocket.send_json({"type": "final_report", "report": runtime.last_report})
                    await websocket.send_json({"type": "system", "message": "Interview stopped.", "ended": True})
                    await websocket.close(code=1000, reason="Interview ended by user")
                    return
                continue

            if kind == "client_transcript":
                if not runtime.accept_client_transcript:
                    continue
                text = str(msg.get("text", "")).strip()
                is_final = bool(msg.get("final", False))
                if text:
                    if is_final:
                        runtime.last_transcript = text
                        DEBUG_RUNTIME["last_final_transcript"] = text
                    else:
                        runtime.last_transcript = text
                        DEBUG_RUNTIME["last_partial_transcript"] = text
                    _debug_record(
                        "client_transcript",
                        {
                            "session_id": runtime.session_id,
                            "final": is_final,
                            "text": text,
                        },
                    )
                    _console_transcript(source="client_ws", text=text, is_final=is_final)
                    _persist_transcript(runtime=runtime, source="client_ws", text=text, is_final=is_final)
                continue

            if kind == "client_transcript_sync":
                if not runtime.accept_client_transcript:
                    continue
                text = str(msg.get("text", "")).strip()
                if text:
                    now_sync = time.time()
                    runtime.last_transcript = text
                    DEBUG_RUNTIME["last_partial_transcript"] = text
                    _debug_record(
                        "client_transcript_sync",
                        {
                            "session_id": runtime.session_id,
                            "text": text,
                        },
                    )
                    should_persist_sync = (
                        (now_sync - float(runtime.last_sync_persist_ts)) >= 0.8 and text != runtime.last_sync_persist_text
                    ) or text.endswith((".", "?", "!"))
                    if should_persist_sync:
                        _console_transcript(source="client_ws_sync", text=text, is_final=False)
                        _persist_transcript(runtime=runtime, source="client_ws_sync", text=text, is_final=False)
                        runtime.last_sync_persist_ts = now_sync
                        runtime.last_sync_persist_text = text
                continue

            if kind == "audio":
                if not runtime.interview_active:
                    continue
                b64 = msg.get("pcm16", "")
                if not b64:
                    continue
                pcm_bytes = base64.b64decode(b64)
                runtime.audio_chunks += 1
                DEBUG_RUNTIME["last_audio_chunk_bytes"] = len(pcm_bytes)
                event = runtime.audio.process_chunk(pcm_bytes)
                runtime.last_audio_total_speaking_duration = float(event.speaking_duration)
                DEBUG_RUNTIME["last_audio_state"] = event.speaking_state
                DEBUG_RUNTIME["last_vad_confidence"] = round(event.vad_confidence, 3)
                question_speaking_duration = max(
                    0.0,
                    float(event.speaking_duration) - float(runtime.question_start_speaking_duration),
                )

                if event.partial_transcript:
                    DEBUG_RUNTIME["last_partial_transcript"] = event.partial_transcript
                    _debug_record(
                        "audio_partial",
                        {
                            "session_id": runtime.session_id,
                            "chunk": runtime.audio_chunks,
                            "state": event.speaking_state,
                            "vad": round(event.vad_confidence, 3),
                            "text": event.partial_transcript,
                        },
                    )
                    _console_transcript(source="whisper_partial", text=event.partial_transcript, is_final=False)
                    _persist_transcript(runtime=runtime, source="whisper_partial", text=event.partial_transcript, is_final=False)
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "text": event.partial_transcript,
                            "final": False,
                        }
                    )

                if event.speech_finished and event.final_pcm16:
                    runtime.pending_transcriptions.append(
                        {
                            "pcm16": event.final_pcm16,
                            "duration": question_speaking_duration,
                            "chunk": runtime.audio_chunks,
                        }
                    )
                    if runtime.transcription_task is None or runtime.transcription_task.done():
                        runtime.transcription_task = asyncio.create_task(_run_transcription_queue(runtime, websocket))

                if event.transcript:
                    runtime.last_transcript = event.transcript
                    DEBUG_RUNTIME["last_final_transcript"] = runtime.last_transcript
                    _debug_record(
                        "audio_final",
                        {
                            "session_id": runtime.session_id,
                            "chunk": runtime.audio_chunks,
                            "duration": round(event.speaking_duration, 2),
                            "text": runtime.last_transcript,
                        },
                    )
                    _console_transcript(source="whisper_final", text=runtime.last_transcript, is_final=True)
                    _persist_transcript(runtime=runtime, source="whisper_final", text=runtime.last_transcript, is_final=True)
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "text": runtime.last_transcript,
                            "final": True,
                        }
                    )

                sm = runtime.state.update_from_audio(
                    {
                        "speaking_state": event.speaking_state,
                        "speaking_duration": question_speaking_duration,
                        "transcript": event.transcript,
                    }
                )

                await websocket.send_json(
                    {
                        "type": "audio_metrics",
                        "speaking_state": event.speaking_state,
                        "speaking_duration": round(question_speaking_duration, 2),
                        "rms": round(event.rms, 4),
                        "vad_confidence": round(event.vad_confidence, 3),
                        "pause_count": runtime.pause_count,
                        "long_silence": runtime.long_silence,
                    }
                )

                runtime.latest_speaking_duration = float(question_speaking_duration)
                if runtime.prev_speaking_state == "speaking" and event.speaking_state == "paused":
                    runtime.pause_count += 1
                    runtime.pause_started_at = time.time()
                    _record_timeline_event(runtime, "silence", max(0.0, 1.0 - event.vad_confidence), {"pause_count": runtime.pause_count})
                    runtime.event_counts["silence"] += 1

                if event.speaking_state == "paused" and runtime.pause_started_at > 0.0:
                    paused_for = time.time() - runtime.pause_started_at
                    if paused_for >= 3.0 and not runtime.long_silence:
                        runtime.long_silence = True
                        runtime.event_counts["long_pause"] += 1
                        _record_timeline_event(runtime, "long_pause", min(1.0, paused_for / 6.0), {"seconds": round(paused_for, 2)})
                if event.speaking_state == "speaking":
                    runtime.pause_started_at = 0.0
                    runtime.long_silence = False

                runtime.prev_speaking_state = event.speaking_state

                if sm["feedback"]:
                    await websocket.send_json({"type": "feedback", "messages": sm["feedback"]})

                now_for_next = time.time()
                question_age = now_for_next - float(runtime.question_started_at)
                min_question_age_seconds = 2.5
                if (
                    "next_question" in sm.get("actions", [])
                    and now_for_next >= runtime.auto_next_cooldown_until
                    and question_age >= min_question_age_seconds
                ):
                    eval_item = await _evaluate_current_answer(runtime)
                    if eval_item:
                        await websocket.send_json(
                            {
                                "type": "qa_analysis_update",
                                "latest": eval_item,
                                "series": runtime.ai_accuracy_series,
                            }
                        )
                    runtime.last_backend_response = runtime.state.generate_response(
                        transcript=runtime.last_transcript,
                        speaking_duration=question_speaking_duration,
                        vision=runtime.latest_vision,
                    )
                    runtime.question_started_at = now_for_next
                    runtime.question_start_speaking_duration = runtime.last_audio_total_speaking_duration
                    runtime.auto_next_cooldown_until = now_for_next + 2.5
                    bundle = await _next_question_bundle(runtime)
                    runtime.active_question_bundle = bundle
                    next_event = runtime.state.reset_for_next_question(bundle.get("question"))
                    await websocket.send_json({"type": "state", **next_event, "ideal_answer": bundle.get("ideal_answer", "")})
                    continue

                is_audio_speaking = event.speaking_state == "speaking"
                is_visual_speaking = bool(runtime.latest_vision.get("visually_speaking", False))
                if is_audio_speaking and not is_visual_speaking:
                    runtime.speech_mismatch_frames += 1
                else:
                    runtime.speech_mismatch_frames = max(0, runtime.speech_mismatch_frames - 1)

                if runtime.speech_mismatch_frames >= 8:
                    await websocket.send_json(
                        {
                            "type": "feedback",
                            "messages": ["Audio speech detected but visible speaking is low. Keep your face in frame and speak clearly."],
                        }
                    )
                    runtime.speech_mismatch_frames = 0

                continue

            if kind == "video":
                if not runtime.interview_active:
                    continue
                b64 = msg.get("jpeg", "")
                if not b64:
                    continue
                jpeg_bytes = base64.b64decode(b64)
                now = time.time()
                _append_frame_for_clip(runtime, now, jpeg_bytes)
                runtime.video_frames += 1
                DEBUG_RUNTIME["last_video_frame_bytes"] = len(jpeg_bytes)
                runtime.latest_vision = runtime.vision.process_frame(jpeg_bytes)
                DEBUG_RUNTIME["last_vision"] = {
                    "eye_contact_score": round(float(runtime.latest_vision.get("eye_contact_score", 0.5)), 3),
                    "head_movement_score": round(float(runtime.latest_vision.get("head_movement_score", 0.5)), 3),
                    "mouth_movement_score": round(float(runtime.latest_vision.get("mouth_movement_score", 0.0)), 3),
                    "visually_speaking": bool(runtime.latest_vision.get("visually_speaking", False)),
                    "gaze_direction": runtime.latest_vision.get("gaze_direction", "unknown"),
                    "detected_objects": runtime.latest_vision.get("detected_objects", []),
                    "baseline_objects": runtime.latest_vision.get("baseline_objects", []),
                }

                eye = float(runtime.latest_vision.get("eye_contact_score", 0.5))
                runtime.eye_samples += 1
                runtime.avg_eye_contact = ((runtime.avg_eye_contact * (runtime.eye_samples - 1)) + eye) / runtime.eye_samples

                gaze_direction = str(runtime.latest_vision.get("gaze_direction", "unknown"))
                if gaze_direction in {"left", "right", "no_face"}:
                    runtime.event_counts["look_away"] += 1
                    _record_timeline_event(runtime, "look_away", 1.0 - eye, {"gaze": gaze_direction})

                suspicious_events = runtime.latest_vision.get("suspicious_events", []) or []
                for ev in suspicious_events:
                    ev_name = str(ev.get("event", ""))
                    ev_conf = float(ev.get("confidence", 0.5))
                    if ev_name in runtime.event_counts:
                        runtime.event_counts[ev_name] += 1
                    _record_timeline_event(runtime, ev_name, ev_conf, {"label": ev.get("label", "")})
                    if ev_name in {"phone_detected", "multiple_persons", "suspicious_movement"}:
                        _trigger_clip(runtime, ev_name, ev_conf, now)

                if runtime.long_silence and gaze_direction in {"left", "right", "no_face"}:
                    runtime.event_counts["multimodal_suspicion"] += 1
                    _record_timeline_event(runtime, "multimodal_suspicion", 0.82, {"reason": "silence + look_away"})

                _flush_completed_clips(runtime, now)

                if (
                    runtime.timeline_version != runtime.last_timeline_sent_version
                    and (now - runtime.last_timeline_emit_ts) >= 0.35
                ):
                    runtime.last_timeline_sent_version = runtime.timeline_version
                    runtime.last_timeline_emit_ts = now
                    await websocket.send_json(
                        {
                            "type": "timeline_update",
                            "timeline": list(runtime.timeline_events)[-20:],
                        }
                    )

                if runtime.video_frames % 8 == 0:
                    _debug_record(
                        "video_metrics",
                        {
                            "session_id": runtime.session_id,
                            "frame": runtime.video_frames,
                            **DEBUG_RUNTIME["last_vision"],
                        },
                    )
                should_emit_vision = (runtime.video_frames % 2 == 0) or bool(runtime.latest_vision.get("alerts", []))
                if should_emit_vision:
                    await websocket.send_json(
                        {
                            "type": "vision_metrics",
                            "eye_contact_score": round(float(runtime.latest_vision.get("eye_contact_score", 0.5)), 3),
                            "head_movement_score": round(float(runtime.latest_vision.get("head_movement_score", 0.5)), 3),
                            "mouth_movement_score": round(float(runtime.latest_vision.get("mouth_movement_score", 0.0)), 3),
                            "visually_speaking": bool(runtime.latest_vision.get("visually_speaking", False)),
                            "gaze_direction": runtime.latest_vision.get("gaze_direction", "unknown"),
                            "detected_objects": runtime.latest_vision.get("detected_objects", []),
                            "baseline_objects": runtime.latest_vision.get("baseline_objects", []),
                            "pause_count": runtime.pause_count,
                            "long_silence": runtime.long_silence,
                            "alerts": runtime.latest_vision.get("alerts", []),
                        }
                    )
                continue

            await asyncio.sleep(0)

    except WebSocketDisconnect:
        if runtime.transcription_task and not runtime.transcription_task.done():
            runtime.transcription_task.cancel()
        SESSION_ANALYTICS[runtime.session_id] = {
            "answer_evaluations": runtime.answer_evaluations,
            "accuracy_series": runtime.ai_accuracy_series,
            "report": runtime.last_report,
            "updated_at": time.time(),
        }
        DEBUG_RUNTIME["connected_clients"] = max(0, int(DEBUG_RUNTIME.get("connected_clients", 1)) - 1)
        _debug_record("session_disconnected", {"session_id": runtime.session_id})
        return
