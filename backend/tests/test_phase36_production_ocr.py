import numpy as np
import pytest

from app.ai import pipeline
from app.ai.detector import Detection
from app.ai.ocr_engines import EngineFragment, EngineResult
from app.ai.pipeline import Candidate, _TimingMetrics, analyze_license_plates
from app.ai.validator import validate_plate_candidate


IMAGE = np.zeros((120, 240, 3), dtype=np.uint8)
DETECTION = Detection(0, "plate_one_line", 0.93, (40, 35, 200, 85))


def engine_result(text, confidence=0.95, fragments=(), noise_detected=False):
    validation = validate_plate_candidate(text)
    return EngineResult(
        "paddleocr",
        text,
        validation.normalized_text,
        confidence,
        validation.is_valid,
        validation.structure_score,
        fragments,
        noise_detected,
    )


def easy_candidate(text, confidence=0.85):
    validation = validate_plate_candidate(text)
    return Candidate(
        raw_text=text,
        normalized_text=validation.normalized_text,
        ocr_confidence=confidence,
        preprocessing_variant="easy_test",
        is_valid=validation.is_valid,
        structure_score=validation.structure_score,
        strategy="full",
        padding_ratio=0.10,
        engine="easyocr",
    )


def install_detection(monkeypatch):
    monkeypatch.setattr(pipeline, "detect_plates", lambda *_args: [DETECTION])


def install_easy_fallback(monkeypatch, text, calls):
    def fake_easy(*_args):
        calls.append("easyocr")
        return [easy_candidate(text)], _TimingMetrics(
            ocr_execution_seconds=0.4,
            ocr_calls=1,
            decoders=["greedy"],
        )

    monkeypatch.setattr(pipeline, "_analyze_standard_normal", fake_easy)


def test_default_production_path_initializes_yolo_before_paddle(monkeypatch):
    events = []

    def detect(*_args):
        events.append("yolo")
        return [DETECTION]

    def paddle(_crop):
        assert events == ["yolo"]
        events.append("paddle")
        return engine_result("29Z138172", 0.94)

    monkeypatch.setattr(pipeline, "PRIMARY_OCR_ENGINE", "paddleocr")
    monkeypatch.setattr(pipeline, "ENABLE_SECONDARY_OCR", True)
    monkeypatch.setattr(pipeline, "detect_plates", detect)
    monkeypatch.setattr(pipeline, "recognize_with_paddleocr", paddle)
    monkeypatch.setattr(
        pipeline,
        "_analyze_standard_normal",
        lambda *_args: (_ for _ in ()).throw(AssertionError("EasyOCR ran")),
    )

    analysis = analyze_license_plates(IMAGE, 0.5)[0]

    assert events == ["yolo", "paddle"]
    assert analysis.winner.normalized_text == "29Z138172"
    assert analysis.fallback_used is False


def test_valid_paddle_result_does_not_call_easyocr(monkeypatch):
    install_detection(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "recognize_with_paddleocr",
        lambda _crop: engine_result("29Z138172", 0.94),
    )
    monkeypatch.setattr(
        pipeline,
        "_analyze_standard_normal",
        lambda *_args: (_ for _ in ()).throw(AssertionError("EasyOCR ran")),
    )
    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]
    assert analysis.winner.normalized_text == "29Z138172"
    assert analysis.ocr_engine == "paddleocr"
    assert analysis.fallback_used is False
    assert analysis.ocr_calls == 1


def test_empty_paddle_result_calls_easyocr(monkeypatch):
    install_detection(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pipeline, "recognize_with_paddleocr", lambda _crop: engine_result("", 0.0)
    )
    install_easy_fallback(monkeypatch, "50G20995", calls)
    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]
    assert calls == ["easyocr"]
    assert analysis.winner.normalized_text == "50G20995"
    assert analysis.fallback_used is True
    assert "empty" in analysis.fallback_reasons


def test_noise_fragments_call_easyocr_even_when_selected_plate_is_valid(monkeypatch):
    install_detection(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pipeline,
        "recognize_with_paddleocr",
        lambda _crop: engine_result("29Z138172", 0.96, noise_detected=True),
    )
    install_easy_fallback(monkeypatch, "29Z138172", calls)

    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]

    assert calls == ["easyocr"]
    assert "noise_fragments" in analysis.fallback_reasons
    assert analysis.winner.strategy == "engine_agreement"


def test_structurally_suspicious_paddle_lets_easyocr_win(monkeypatch):
    install_detection(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pipeline,
        "recognize_with_paddleocr",
        lambda _crop: engine_result("638957926", 0.99),
    )
    install_easy_fallback(monkeypatch, "63B957926", calls)
    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]
    assert calls == ["easyocr"]
    assert analysis.winner.normalized_text == "63B957926"
    assert analysis.winner.engine == "easyocr"


def test_engine_agreement_is_strong_evidence(monkeypatch):
    install_detection(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pipeline,
        "recognize_with_paddleocr",
        lambda _crop: engine_result("71B158609", 0.50),
    )
    install_easy_fallback(monkeypatch, "71B158609", calls)
    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]
    assert analysis.winner.normalized_text == "71B158609"
    assert analysis.winner.strategy == "engine_agreement"
    assert analysis.winner.engine == "paddleocr+easyocr"


def test_engine_disagreement_uses_structure_not_raw_confidence(monkeypatch):
    install_detection(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pipeline,
        "recognize_with_paddleocr",
        lambda _crop: engine_result("638957926", 0.999),
    )
    install_easy_fallback(monkeypatch, "63B957926", calls)
    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]
    assert analysis.winner.normalized_text == "63B957926"
    assert analysis.winner.ocr_confidence < 0.999


def test_both_engines_missing_series_letter_remains_uncertain(monkeypatch):
    install_detection(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pipeline,
        "recognize_with_paddleocr",
        lambda _crop: engine_result("51081343", 0.996),
    )
    install_easy_fallback(monkeypatch, "51081343", calls)
    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]
    assert analysis.winner.normalized_text == "51081343"
    assert analysis.winner.is_valid is False
    assert analysis.ocr_status == "unreadable"


@pytest.mark.parametrize(
    "text",
    ("63B957926", "50G20995", "29Z138172", "29S668708", "51D81343", "71B158609"),
)
def test_known_regression_shapes_are_structurally_usable(text):
    candidate = easy_candidate(text)
    assert pipeline.paddle_fallback_reasons(candidate) == ()


def test_paddle_fragments_participate_in_two_line_row_fusion(monkeypatch):
    two_line = Detection(1, "plate_two_line", 0.94, (60, 10, 160, 100))
    monkeypatch.setattr(pipeline, "detect_plates", lambda *_args: [two_line])
    fragments = (
        EngineFragment("29Z1", 0.95, (), (40.0, 20.0)),
        EngineFragment("58344", 0.97, (), (40.0, 70.0)),
    )
    monkeypatch.setattr(
        pipeline,
        "recognize_with_paddleocr",
        lambda _crop: engine_result("29Z158344", 0.96, fragments),
    )
    analysis = analyze_license_plates(IMAGE, 0.5, engine="production")[0]
    assert analysis.winner.normalized_text == "29Z158344"
    assert analysis.top_row_text == "29Z1"
    assert analysis.bottom_row_text == "58344"
    assert analysis.fallback_used is False
