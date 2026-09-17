from dataclasses import asdict, dataclass

import numpy as np

from app.ai.detector import Detection, detect_plates
from app.ai.ocr import run_ocr
from app.ai.preprocess import crop_with_padding, preprocessing_variants
from app.ai.validator import validate_plate_candidate


@dataclass(frozen=True)
class Candidate:
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    preprocessing_variant: str
    is_valid: bool


def _ocr_detection(image: np.ndarray, detection: Detection) -> dict[str, object]:
    crop = crop_with_padding(image, detection.bbox)
    candidates: list[Candidate] = []
    for variant_name, variant_image in preprocessing_variants(crop).items():
        ocr_result = run_ocr(variant_image)
        validation = validate_plate_candidate(ocr_result.raw_text)
        candidates.append(
            Candidate(
                raw_text=ocr_result.raw_text,
                normalized_text=validation.normalized_text,
                ocr_confidence=ocr_result.confidence,
                preprocessing_variant=variant_name,
                is_valid=validation.is_valid,
            )
        )

    winner = max(candidates, key=lambda item: (item.is_valid, item.ocr_confidence), default=None)
    if winner is None:
        winner = Candidate("", "", 0.0, "none", False)
    x1, y1, x2, y2 = detection.bbox
    return {
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "detection_confidence": detection.confidence,
        **asdict(winner),
    }


def _ranking_score(result: dict[str, object]) -> tuple[bool, float]:
    combined = (0.65 * float(result["ocr_confidence"])) + (
        0.35 * float(result["detection_confidence"])
    )
    return bool(result["is_valid"]), combined


def recognize_license_plates(image: np.ndarray, confidence_threshold: float) -> list[dict[str, object]]:
    detections = detect_plates(image, confidence_threshold)
    results = [_ocr_detection(image, detection) for detection in detections]
    results.sort(key=_ranking_score, reverse=True)
    return results
