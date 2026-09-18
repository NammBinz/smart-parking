from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from app.ai.detector import Detection, detect_plates
from app.ai.preprocess import crop_to_bbox
from app.ai.quality import assess_detection_quality


CAMERA_PREFERRED_WIDTH = 60
CAMERA_PREFERRED_HEIGHT = 24
CAMERA_PREFERRED_AREA = 1800
CAMERA_MIN_SHARPNESS = 25.0
CAPTURE_ZONE = (0.15, 0.25, 0.85, 0.80)


@dataclass(frozen=True)
class FrameQuality:
    detection: Detection
    width: int
    height: int
    area: int
    sharpness: float
    brightness: float
    brightness_quality: float
    size_quality: float
    center_bonus: float
    quality_score: float
    camera_status: str


@dataclass(frozen=True)
class FrameQualityAnalysis:
    detections: tuple[FrameQuality, ...]
    yolo_seconds: float
    quality_seconds: float


def calculate_sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return max(0.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()))


def calculate_brightness_quality(crop: np.ndarray) -> tuple[float, float]:
    if crop.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    brightness = float(gray.mean())
    quality = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    return brightness, quality


def calculate_size_quality(width: int, height: int, area: int) -> float:
    components = (
        min(1.0, width / 160.0),
        min(1.0, height / 60.0),
        min(1.0, area / 9600.0),
    )
    return sum(components) / len(components)


def center_capture_bonus(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    zone: tuple[float, float, float, float] = CAPTURE_ZONE,
) -> float:
    if image_width <= 0 or image_height <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2) / image_width
    center_y = ((y1 + y2) / 2) / image_height
    return 1.0 if zone[0] <= center_x <= zone[2] and zone[1] <= center_y <= zone[3] else 0.0


def calculate_frame_quality_score(
    detection_confidence: float,
    size_quality: float,
    sharpness_quality: float,
    center_bonus: float,
    brightness_quality: float,
) -> float:
    """Blend bounded factors; raw sharpness is normalized by the caller/sequence."""
    score = (
        0.28 * max(0.0, min(1.0, detection_confidence))
        + 0.25 * max(0.0, min(1.0, size_quality))
        + 0.25 * max(0.0, min(1.0, sharpness_quality))
        + 0.12 * max(0.0, min(1.0, center_bonus))
        + 0.10 * max(0.0, min(1.0, brightness_quality))
    )
    return max(0.0, min(1.0, score))


def _camera_status(
    detection: Detection, width: int, height: int, area: int, sharpness: float
) -> str:
    if not assess_detection_quality(detection).should_run:
        return "too_small"
    if (
        width < CAMERA_PREFERRED_WIDTH
        or height < CAMERA_PREFERRED_HEIGHT
        or area < CAMERA_PREFERRED_AREA
    ):
        return "move_closer"
    if sharpness < CAMERA_MIN_SHARPNESS:
        return "too_blurry"
    return "candidate"


def analyze_frame_quality(image: np.ndarray, confidence_threshold: float) -> FrameQualityAnalysis:
    yolo_started = perf_counter()
    detections = detect_plates(image, confidence_threshold)
    yolo_seconds = perf_counter() - yolo_started
    quality_started = perf_counter()
    image_height, image_width = image.shape[:2]
    results: list[FrameQuality] = []

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        area = width * height
        crop = crop_to_bbox(image, detection.bbox)
        sharpness = calculate_sharpness(crop)
        brightness, brightness_quality = calculate_brightness_quality(crop)
        size_quality = calculate_size_quality(width, height, area)
        center_bonus = center_capture_bonus(
            detection.bbox, image_width, image_height
        )
        # This provisional transform is only for a single-frame ranking. The
        # camera client recomputes sharpness relatively across its short buffer.
        sharpness_quality = sharpness / (sharpness + 120.0) if sharpness else 0.0
        score = calculate_frame_quality_score(
            detection.confidence,
            size_quality,
            sharpness_quality,
            center_bonus,
            brightness_quality,
        )
        results.append(
            FrameQuality(
                detection=detection,
                width=width,
                height=height,
                area=area,
                sharpness=sharpness,
                brightness=brightness,
                brightness_quality=brightness_quality,
                size_quality=size_quality,
                center_bonus=center_bonus,
                quality_score=score,
                camera_status=_camera_status(detection, width, height, area, sharpness),
            )
        )

    return FrameQualityAnalysis(
        detections=tuple(results),
        yolo_seconds=yolo_seconds,
        quality_seconds=perf_counter() - quality_started,
    )
