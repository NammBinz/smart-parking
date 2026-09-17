import threading
from dataclasses import dataclass

import numpy as np

from app.ai.model_manager import get_yolo_model


_inference_lock = threading.Lock()


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detect_plates(image: np.ndarray, confidence_threshold: float) -> list[Detection]:
    height, width = image.shape[:2]
    model = get_yolo_model()
    with _inference_lock:
        results = model.predict(
            source=image,
            conf=float(confidence_threshold),
            imgsz=640,
            verbose=False,
        )

    detections: list[Detection] = []
    if not results:
        return detections

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        coordinates = box.xyxy[0].detach().cpu().tolist()
        x1 = max(0, min(width, int(round(coordinates[0]))))
        y1 = max(0, min(height, int(round(coordinates[1]))))
        x2 = max(0, min(width, int(round(coordinates[2]))))
        y2 = max(0, min(height, int(round(coordinates[3]))))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(box.cls[0].detach().cpu().item())
        detection_confidence = float(box.conf[0].detach().cpu().item())
        detections.append(
            Detection(
                class_id=class_id,
                class_name=_class_name(model.names, class_id),
                confidence=max(0.0, min(1.0, detection_confidence)),
                bbox=(x1, y1, x2, y2),
            )
        )
    return detections
