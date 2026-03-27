from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Detection:
    label: str
    confidence: float
    area_ratio: float
    center_xy: Tuple[float, float]
    box_xyxy: Tuple[float, float, float, float]


class YoloDetector:
    """Thin YOLOv8 wrapper optimized for CPU inference."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, imgsz: int = 320) -> None:
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._model = None

        try:
            from ultralytics import YOLO

            resolved = Path(model_path)
            self._model = YOLO(str(resolved))
        except Exception:
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr) -> List[Detection]:
        if self._model is None:
            return []

        detections: List[Detection] = []
        try:
            results = self._model.predict(
                source=frame_bgr,
                verbose=False,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device="cpu",
            )
            if not results:
                return []

            result = results[0]
            names = result.names
            boxes = result.boxes
            if boxes is None:
                return []

            h, w = frame_bgr.shape[:2]
            frame_area = float(max(1, h * w))

            for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                label = names.get(int(cls_id), str(int(cls_id)))
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                box_w = max(1.0, x2 - x1)
                box_h = max(1.0, y2 - y1)
                area_ratio = float((box_w * box_h) / frame_area)
                cx = float((x1 + x2) * 0.5 / max(1.0, float(w)))
                cy = float((y1 + y2) * 0.5 / max(1.0, float(h)))
                detections.append(
                    Detection(
                        label=str(label),
                        confidence=float(conf),
                        area_ratio=area_ratio,
                        center_xy=(cx, cy),
                        box_xyxy=(x1, y1, x2, y2),
                    )
                )
        except Exception:
            return []

        return detections
