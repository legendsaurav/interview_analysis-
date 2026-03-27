from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .yolo_model import YoloDetector

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency
    mp = None


class VisionProcessor:
    """Face behavior + sparse YOLO object checks with frame skipping."""

    def __init__(
        self,
        yolo_model_path: str,
        yolo_every_n_frames: int = 4,
        baseline_cycles: int = 3,
    ) -> None:
        self.frame_index = 0
        self.yolo_every_n_frames = max(1, yolo_every_n_frames)
        self.detector = YoloDetector(str(Path(yolo_model_path)))
        self.baseline_cycles = max(1, baseline_cycles)
        self._baseline_remaining = self.baseline_cycles
        self._baseline_counts: Dict[str, int] = defaultdict(int)
        self._baseline_objects: set[str] = set()
        self._track_state: Dict[str, Dict[str, Any]] = {}

        self._last_nose_xy: Optional[np.ndarray] = None
        self._last_mouth_ratio: Optional[float] = None
        self._last_objects: List[str] = []
        self._last_suspicious_events: List[Dict[str, Any]] = []

        self._face_mesh = None
        if mp is not None:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._face_mesh = None

    def process_frame(self, jpeg_bytes: bytes) -> Dict[str, object]:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return self._empty_result()

        self.frame_index += 1
        face_metrics = self._analyze_face(frame)

        if self.frame_index % self.yolo_every_n_frames == 0:
            detections = self.detector.detect(frame)
            self._last_objects, self._last_suspicious_events = self._filter_detections(detections)

        alerts = []
        for ev in self._last_suspicious_events:
            if ev["event"] == "multiple_persons":
                alerts.append("Multiple persons detected")
            elif ev["event"] == "phone_detected":
                alerts.append("Phone usage detected")
            elif ev["event"] == "suspicious_movement":
                alerts.append("Suspicious movement detected")

        return {
            **face_metrics,
            "detected_objects": self._last_objects,
            "alerts": alerts,
            "baseline_objects": sorted(self._baseline_objects),
            "suspicious_events": self._last_suspicious_events,
        }

    def _empty_result(self) -> Dict[str, object]:
        return {
            "eye_contact_score": 0.5,
            "head_movement_score": 0.5,
            "gaze_direction": "unknown",
            "mouth_movement_score": 0.0,
            "visually_speaking": False,
            "detected_objects": self._last_objects,
            "alerts": [],
            "baseline_objects": sorted(self._baseline_objects),
            "suspicious_events": [],
        }

    def _filter_detections(self, detections) -> Tuple[List[str], List[Dict[str, Any]]]:
        # Filter 1: confidence + size + central region filtering.
        filtered = []
        for d in detections:
            if d.confidence < 0.5:
                continue
            if d.area_ratio < 0.006:
                continue
            cx, cy = d.center_xy
            if not (0.12 <= cx <= 0.88 and 0.12 <= cy <= 0.88):
                continue
            filtered.append(d)

        if self._baseline_remaining > 0:
            for d in filtered:
                self._baseline_counts[d.label] += 1
            self._baseline_remaining -= 1
            if self._baseline_remaining == 0:
                self._baseline_objects = {k for k, v in self._baseline_counts.items() if v >= 2}
            return sorted({d.label for d in filtered}), []

        # Keep most confident detection per label for lightweight tracking.
        best_by_label: Dict[str, Any] = {}
        for d in filtered:
            existing = best_by_label.get(d.label)
            if existing is None or d.confidence > existing.confidence:
                best_by_label[d.label] = d

        suspicious_events: List[Dict[str, Any]] = []
        now_seen: set[str] = set()
        valid_labels: List[str] = []

        for label, d in best_by_label.items():
            now_seen.add(label)
            prev = self._track_state.get(label)
            moved = False
            if prev and "center_xy" in prev:
                px, py = prev["center_xy"]
                moved = float(np.linalg.norm(np.array([d.center_xy[0] - px, d.center_xy[1] - py]))) > 0.06

            if prev and prev.get("seen_last", False):
                consecutive = int(prev.get("consecutive", 0)) + 1
            else:
                consecutive = 1

            moving_count = int(prev.get("moving_count", 0) if prev else 0)
            if moved:
                moving_count += 1
            else:
                moving_count = max(0, moving_count - 1)

            self._track_state[label] = {
                "center_xy": d.center_xy,
                "consecutive": consecutive,
                "moving_count": moving_count,
                "seen_last": True,
            }

            # Filter 2 + 3: baseline awareness + temporal persistence.
            is_baseline = label in self._baseline_objects
            is_persistent = consecutive >= 3
            is_moving = moving_count >= 2

            if is_baseline and not is_moving:
                continue
            if not is_persistent:
                continue

            valid_labels.append(label)

            # Filter 4: movement check integrated into event decision.
            if "phone" in label.lower() and (is_moving or not is_baseline):
                suspicious_events.append({"event": "phone_detected", "confidence": d.confidence, "label": label})

            if label == "person":
                person_count = sum(1 for x in filtered if x.label == "person")
                if person_count > 1:
                    suspicious_events.append({"event": "multiple_persons", "confidence": d.confidence, "label": label})

            if is_moving and label not in {"person"}:
                suspicious_events.append({"event": "suspicious_movement", "confidence": d.confidence, "label": label})

        for label, state in self._track_state.items():
            if label not in now_seen:
                state["seen_last"] = False
                state["consecutive"] = 0

        return sorted(set(valid_labels)), suspicious_events

    def _analyze_face(self, frame_bgr) -> Dict[str, object]:
        if self._face_mesh is None:
            return {
                "eye_contact_score": 0.5,
                "head_movement_score": 0.5,
                "gaze_direction": "unknown",
                "mouth_movement_score": 0.0,
                "visually_speaking": False,
            }

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            self._last_nose_xy = None
            self._last_mouth_ratio = None
            return {
                "eye_contact_score": 0.2,
                "head_movement_score": 0.5,
                "gaze_direction": "no_face",
                "mouth_movement_score": 0.0,
                "visually_speaking": False,
            }

        lm = result.multi_face_landmarks[0].landmark
        left_eye = np.array([(lm[33].x + lm[133].x) / 2.0, (lm[33].y + lm[133].y) / 2.0])
        right_eye = np.array([(lm[362].x + lm[263].x) / 2.0, (lm[362].y + lm[263].y) / 2.0])
        nose = np.array([lm[1].x, lm[1].y])
        eye_mid = (left_eye + right_eye) / 2.0

        yaw = float(nose[0] - eye_mid[0])
        pitch = float(nose[1] - eye_mid[1])

        if yaw < -0.03:
            gaze_direction = "left"
        elif yaw > 0.03:
            gaze_direction = "right"
        else:
            gaze_direction = "center"

        eye_contact_score = float(np.clip(1.0 - abs(yaw) * 12.0, 0.0, 1.0))

        if self._last_nose_xy is None:
            movement = 0.0
        else:
            movement = float(np.linalg.norm(nose - self._last_nose_xy))
        self._last_nose_xy = nose
        head_movement_score = float(np.clip(1.0 - movement * 40.0, 0.0, 1.0))

        # Include a basic roll proxy derived from vertical eye disparity.
        roll_proxy = float(left_eye[1] - right_eye[1])
        if abs(roll_proxy) > 0.03:
            head_movement_score = max(0.0, head_movement_score - 0.1)

        # Visual speaking proxy from lip opening normalized by face scale.
        upper_lip = np.array([lm[13].x, lm[13].y])
        lower_lip = np.array([lm[14].x, lm[14].y])
        mouth_gap = float(np.linalg.norm(lower_lip - upper_lip))
        face_scale = float(np.linalg.norm(right_eye - left_eye) + 1e-6)
        mouth_ratio = mouth_gap / face_scale

        if self._last_mouth_ratio is None:
            mouth_delta = 0.0
        else:
            mouth_delta = abs(mouth_ratio - self._last_mouth_ratio)
        self._last_mouth_ratio = mouth_ratio

        mouth_movement_score = float(np.clip((mouth_ratio * 4.5) + (mouth_delta * 22.0), 0.0, 1.0))
        visually_speaking = mouth_movement_score >= 0.38

        _ = pitch  # Pitch is computed for potential downstream rules.

        return {
            "eye_contact_score": eye_contact_score,
            "head_movement_score": head_movement_score,
            "gaze_direction": gaze_direction,
            "mouth_movement_score": mouth_movement_score,
            "visually_speaking": visually_speaking,
        }
