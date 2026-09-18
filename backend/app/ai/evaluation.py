from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean


def levenshtein_distance(expected: str, predicted: str) -> int:
    previous = list(range(len(predicted) + 1))
    for expected_index, expected_character in enumerate(expected, start=1):
        current = [expected_index]
        for predicted_index, predicted_character in enumerate(predicted, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[predicted_index] + 1,
                    previous[predicted_index - 1]
                    + (expected_character != predicted_character),
                )
            )
        previous = current
    return previous[-1]


def character_accuracy(expected: str, predicted: str) -> float:
    width = max(len(expected), len(predicted), 1)
    return max(0.0, 1.0 - (levenshtein_distance(expected, predicted) / width))


def classify_error(
    expected: str,
    predicted: str,
    *,
    detected: bool,
    ocr_status: str = "",
    quality_issue: str = "",
) -> str:
    if not detected:
        return "detector_miss"
    if ocr_status == "too_small":
        return "too_small"
    if expected == predicted:
        return "exact"
    if quality_issue in {"blur", "perspective"}:
        return quality_issue
    if not predicted:
        return "unreadable"
    if len(predicted) < len(expected):
        return "missing_characters"
    if len(predicted) > len(expected):
        return "extra_characters"
    return "substitution_or_order"


@dataclass(frozen=True)
class EvaluationRow:
    filename: str
    expected: str
    predicted: str
    detected: bool
    exact_match: bool
    edit_distance: int
    character_accuracy: float
    error_category: str
    bbox: tuple[int, int, int, int] | None = None
    raw_ocr: str = ""
    ocr_valid: bool = False
    detection_confidence: float = 0.0
    ocr_confidence: float = 0.0
    ocr_status: str = ""
    yolo_seconds: float = 0.0
    quality_seconds: float = 0.0
    ocr_seconds: float = 0.0
    total_seconds: float = 0.0
    engine: str = "easyocr"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def make_evaluation_row(
    *,
    filename: str,
    expected: str,
    predicted: str,
    detected: bool,
    detection_confidence: float = 0.0,
    ocr_confidence: float = 0.0,
    ocr_status: str = "",
    yolo_seconds: float = 0.0,
    quality_seconds: float = 0.0,
    ocr_seconds: float = 0.0,
    total_seconds: float = 0.0,
    engine: str = "easyocr",
    quality_issue: str = "",
    bbox: tuple[int, int, int, int] | None = None,
    raw_ocr: str = "",
    ocr_valid: bool = False,
) -> EvaluationRow:
    distance = levenshtein_distance(expected, predicted)
    return EvaluationRow(
        filename=filename,
        expected=expected,
        predicted=predicted,
        detected=detected,
        exact_match=expected == predicted,
        edit_distance=distance,
        character_accuracy=character_accuracy(expected, predicted),
        error_category=classify_error(
            expected,
            predicted,
            detected=detected,
            ocr_status=ocr_status,
            quality_issue=quality_issue,
        ),
        bbox=bbox,
        raw_ocr=raw_ocr,
        ocr_valid=ocr_valid,
        detection_confidence=detection_confidence,
        ocr_confidence=ocr_confidence,
        ocr_status=ocr_status,
        yolo_seconds=yolo_seconds,
        quality_seconds=quality_seconds,
        ocr_seconds=ocr_seconds,
        total_seconds=total_seconds,
        engine=engine,
    )


def summarize_evaluations(rows: list[EvaluationRow]) -> dict[str, object]:
    total = len(rows)
    detected = [row for row in rows if row.detected]
    exact = [row for row in rows if row.exact_match]
    valid_wrong = [row for row in rows if row.ocr_valid and not row.exact_match]
    invalid = [
        row
        for row in rows
        if row.detected and not row.ocr_valid and row.ocr_status != "too_small"
    ]
    too_small = [row for row in rows if row.ocr_status == "too_small"]
    return {
        "images": total,
        "detected": len(detected),
        "detection_rate": len(detected) / total if total else 0.0,
        "exact_matches": len(exact),
        "valid_but_wrong": len(valid_wrong),
        "invalid_ocr": len(invalid),
        "too_small": len(too_small),
        "exact_plate_accuracy": len(exact) / total if total else 0.0,
        "ocr_exact_match_rate_detected": len(exact) / len(detected) if detected else 0.0,
        "mean_character_accuracy": mean(row.character_accuracy for row in rows) if rows else 0.0,
        "mean_detection_confidence": mean(row.detection_confidence for row in detected) if detected else 0.0,
        "mean_ocr_confidence": mean(row.ocr_confidence for row in detected) if detected else 0.0,
        "mean_yolo_seconds": mean(row.yolo_seconds for row in rows) if rows else 0.0,
        "mean_quality_seconds": mean(row.quality_seconds for row in rows) if rows else 0.0,
        "mean_ocr_seconds": mean(row.ocr_seconds for row in rows) if rows else 0.0,
        "mean_total_seconds": mean(row.total_seconds for row in rows) if rows else 0.0,
        "mean_inference_seconds": mean(row.total_seconds for row in rows) if rows else 0.0,
        "error_categories": dict(Counter(row.error_category for row in rows)),
    }
