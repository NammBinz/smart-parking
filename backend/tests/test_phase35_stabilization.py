import cv2
import numpy as np

from app.ai import ocr_engines
from app.ai.evaluation import (
    character_accuracy,
    classify_error,
    levenshtein_distance,
    make_evaluation_row,
    summarize_evaluations,
)
from app.ai.ocr_engines import EngineResult, recognize_with_optional_fallback
from app.ai.rectification import rectify_plate_crop


def test_edit_distance_and_character_accuracy():
    assert levenshtein_distance("29Z158344", "29Z15834") == 1
    assert character_accuracy("29Z158344", "29Z15834") == 8 / 9
    assert character_accuracy("ABC", "") == 0.0


def test_evaluation_separates_detector_and_ocr_results():
    exact = make_evaluation_row(
        filename="exact.jpg",
        expected="51A4032",
        predicted="51A4032",
        detected=True,
        ocr_status="ok",
        ocr_valid=True,
    )
    miss = make_evaluation_row(
        filename="miss.jpg", expected="29Z158344", predicted="", detected=False
    )
    wrong = make_evaluation_row(
        filename="wrong.jpg",
        expected="29Z158344",
        predicted="29Z15834",
        detected=True,
        ocr_status="ok",
        ocr_valid=True,
    )
    summary = summarize_evaluations([exact, miss, wrong])
    assert summary["detection_rate"] == 2 / 3
    assert summary["exact_plate_accuracy"] == 1 / 3
    assert summary["ocr_exact_match_rate_detected"] == 1 / 2
    assert summary["valid_but_wrong"] == 1
    assert summary["invalid_ocr"] == 0
    assert classify_error("ABC", "AB", detected=True) == "missing_characters"


def test_secondary_ocr_is_disabled_by_default_and_bounded_when_enabled(monkeypatch):
    calls = []

    def primary(_image):
        calls.append("primary")
        return EngineResult("easyocr", "BAD", "BAD", 0.2, False)

    def secondary(_image):
        calls.append("secondary")
        return EngineResult("paddleocr", "51A4032", "51A4032", 0.9, True)

    image = np.zeros((40, 120, 3), dtype=np.uint8)
    monkeypatch.setattr(ocr_engines, "ENABLE_SECONDARY_OCR", False)
    first = recognize_with_optional_fallback(image, primary=primary, secondary=secondary)
    assert first.engine == "easyocr"
    assert calls == ["primary"]

    calls.clear()
    monkeypatch.setattr(ocr_engines, "ENABLE_SECONDARY_OCR", True)
    chosen = recognize_with_optional_fallback(
        image, primary=primary, secondary=secondary
    )
    assert chosen.engine == "paddleocr"
    assert calls == ["primary", "secondary"]


def test_rectification_has_safe_fallback_and_corrects_a_trapezoid():
    blank = np.full((80, 180, 3), 127, dtype=np.uint8)
    unchanged, applied = rectify_plate_crop(blank)
    assert applied is False
    assert unchanged is blank

    trapezoid = np.zeros((140, 260, 3), dtype=np.uint8)
    points = np.array([[35, 30], [225, 15], [205, 115], [55, 120]], dtype=np.int32)
    cv2.polylines(trapezoid, [points], True, (255, 255, 255), 4)
    rectified, applied = rectify_plate_crop(trapezoid)
    assert applied is True
    assert rectified.shape[1] > rectified.shape[0]
