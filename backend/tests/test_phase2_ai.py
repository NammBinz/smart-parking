from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.ai import pipeline as pipeline_module
from app.ai import ocr as ocr_module
from app.ai.detector import Detection
from app.ai.ocr import OCRResult, run_ocr, sort_fragments_spatially
from app.ai.pipeline import Candidate, analyze_license_plates, select_best_candidate
from app.ai.preprocess import crop_with_padding, preprocessing_variants
from app.ai.validator import plate_structure_score, validate_plate_candidate
from app.api import ai as ai_api
from app.database import SessionLocal
from app.main import app
from app.models import ParkingSession
from scripts import test_recognition as recognition_script


TEST_DB = Path(__file__).parent / "test_parking.db"


def _png_bytes() -> bytes:
    image = np.full((40, 80, 3), 180, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _detection(text="29A17938", valid=True, confidence=0.92):
    return {
        "class_id": 0,
        "class_name": "license_plate",
        "bbox": {"x1": 4, "y1": 5, "x2": 65, "y2": 30},
        "detection_confidence": 0.96,
        "raw_text": text,
        "normalized_text": text,
        "ocr_confidence": confidence,
        "preprocessing_variant": "adaptive_threshold",
        "is_valid": valid,
    }


def test_invalid_and_unsupported_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_api, "UPLOAD_DIR", tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/ai/recognize-image").status_code == 400
        unsupported = client.post(
            "/api/ai/recognize-image",
            files={"file": ("plate.gif", b"GIF89a", "image/gif")},
        )
        assert unsupported.status_code == 400
        invalid = client.post(
            "/api/ai/recognize-image",
            files={"file": ("plate.jpg", b"\xff\xd8\xffnot-an-image", "image/jpeg")},
        )
        assert invalid.status_code == 400
        oversized = client.post(
            "/api/ai/recognize-image",
            files={
                "file": (
                    "plate.png",
                    b"\x89PNG\r\n\x1a\n" + b"0" * (ai_api.MAX_UPLOAD_BYTES + 1),
                    "image/png",
                )
            },
        )
        assert oversized.status_code == 400


def test_success_no_detection_multiple_and_invalid_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_api, "UPLOAD_DIR", tmp_path)
    responses = [
        [_detection()],
        [],
        [_detection("30G12345", True, 0.88), _detection("BAD", False, 0.2)],
        [_detection("BAD", False, 0.15)],
    ]

    thresholds = []

    def mocked_pipeline(_image, threshold):
        thresholds.append(threshold)
        return responses.pop(0)

    monkeypatch.setattr(ai_api, "recognize_license_plates", mocked_pipeline)
    with TestClient(app) as client:
        success = client.post(
            "/api/ai/recognize-image",
            files={"file": ("vehicle.png", _png_bytes(), "image/png")},
        )
        assert success.status_code == 200
        assert success.json()["recognized"] is True
        assert success.json()["detections"][0]["normalized_text"] == "29A17938"
        assert success.json()["image_path"].startswith("/uploads/")

        none = client.post(
            "/api/ai/recognize-image",
            files={"file": ("vehicle.png", _png_bytes(), "image/png")},
        )
        assert none.status_code == 200
        assert none.json()["recognized"] is False
        assert none.json()["best_index"] is None

        multiple = client.post(
            "/api/ai/recognize-image",
            files={"file": ("vehicle.png", _png_bytes(), "image/png")},
        )
        assert multiple.status_code == 200
        assert len(multiple.json()["detections"]) == 2

        invalid_ocr = client.post(
            "/api/ai/recognize-image",
            files={"file": ("vehicle.png", _png_bytes(), "image/png")},
        )
        assert invalid_ocr.status_code == 200
        assert invalid_ocr.json()["detections"][0]["is_valid"] is False
        assert thresholds == [0.25, 0.25, 0.25, 0.25]
        assert all(path.suffix == ".png" for path in tmp_path.iterdir())


def test_status_and_ai_check_in_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_api,
        "runtime_status",
        lambda: {
            "model_available": True,
            "model_path": "backend/models/best.pt",
            "model_names": {"0": "plate", "1": "plate_two_line"},
            "ocr_available": True,
        },
    )
    with TestClient(app) as client:
        status_response = client.get("/api/ai/status")
        assert status_response.status_code == 200
        assert set(status_response.json()["model_names"]) == {"0", "1"}

        slots = client.get("/api/slots").json()
        ai_check_in = client.post(
            "/api/parking/check-in",
            json={
                "plate_number": "29A17938",
                "slot_id": next(slot["id"] for slot in slots if slot["code"] == "A1"),
                "entry_image": "/uploads/test-image.jpg",
                "entry_detection_confidence": 0.96,
                "entry_ocr_confidence": 0.93,
            },
        )
        assert ai_check_in.status_code == 201

        manual_check_in = client.post(
            "/api/parking/check-in",
            json={
                "plate_number": "51-F1 222.33",
                "slot_id": next(slot["id"] for slot in slots if slot["code"] == "A2"),
            },
        )
        assert manual_check_in.status_code == 201

    with SessionLocal() as db:
        ai_session = db.scalar(
            select(ParkingSession).where(ParkingSession.plate_number == "29A17938")
        )
        assert ai_session.entry_image == "/uploads/test-image.jpg"
        assert ai_session.entry_detection_confidence == 0.96
        assert ai_session.entry_ocr_confidence == 0.93
        manual_session = db.scalar(
            select(ParkingSession).where(ParkingSession.plate_number == "51F122233")
        )
        assert manual_session.entry_image is None


def test_preprocessing_validation_and_two_line_ordering():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_with_padding(image, (0, 0, 50, 30))
    assert crop.shape[:2] == (32, 52)
    variants = preprocessing_variants(crop)
    assert set(variants) == {
        "upscaled_color",
        "grayscale",
        "contrast_enhanced",
        "adaptive_threshold",
        "otsu",
        "bilateral_otsu",
        "sharpened_grayscale",
    }

    fragments = [
        ([[0, 30], [20, 30], [20, 45], [0, 45]], "184", 0.8),
        ([[25, 30], [40, 30], [40, 45], [25, 45]], "61", 0.9),
        ([[25, 0], [40, 0], [40, 15], [25, 15]], "F1", 0.95),
        ([[0, 0], [20, 0], [20, 15], [0, 15]], "59", 0.9),
    ]
    ordered = sort_fragments_spatially(fragments)
    assert "".join(fragment[1] for fragment in ordered) == "59F118461"
    assert validate_plate_candidate("29-A1 179.38").normalized_text == "29A117938"
    assert validate_plate_candidate("29-A1 179.38").is_valid is True
    assert validate_plate_candidate("BAD").is_valid is False


def test_multiple_padding_and_two_line_split_candidates(monkeypatch, tmp_path, capsys):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = Detection(
        class_id=1,
        class_name="1",
        confidence=0.93,
        bbox=(20, 20, 80, 80),
    )
    monkeypatch.setattr(pipeline_module, "detect_plates", lambda _image, _confidence: [detection])

    def mocked_ocr(_image, source_region="full"):
        if source_region == "top":
            return OCRResult(raw_text="59F1", confidence=0.85)
        if source_region == "bottom":
            return OCRResult(raw_text="18461", confidence=0.88)
        return OCRResult(raw_text="292160344", confidence=0.95)

    monkeypatch.setattr(pipeline_module, "run_ocr", mocked_ocr)
    analysis = analyze_license_plates(image, 0.25)[0]
    assert analysis.debug_images == {}
    identities = {candidate.preprocessing_variant for candidate in analysis.candidates}
    assert any(identity.startswith("padding05_") for identity in identities)
    assert any(identity.startswith("padding10_") for identity in identities)
    assert any(identity.startswith("padding15_") for identity in identities)
    assert "padding10_split_rows_otsu" in identities
    assert analysis.split_row_used is True
    assert analysis.winner.strategy == "split_rows"
    assert analysis.winner.normalized_text == "59F118461"
    assert any(candidate.raw_text == "292160344" for candidate in analysis.candidates)

    exhaustive = analyze_license_plates(image, 0.25, exhaustive=True, capture_images=True)[0]
    assert len(exhaustive.candidates) == 42
    assert "padding05_otsu" in {
        candidate.preprocessing_variant for candidate in exhaustive.candidates
    }
    assert "padding15_split_rows_sharpened_grayscale" in {
        candidate.preprocessing_variant for candidate in exhaustive.candidates
    }
    assert "padding10_crop" in exhaustive.debug_images
    monkeypatch.setattr(recognition_script, "DEBUG_OUTPUT_DIR", tmp_path)
    recognition_script._print_debug_analysis(1, exhaustive)
    output = capsys.readouterr().out
    assert "OCR candidates:" in output
    assert "raw: 292160344" in output
    assert "Winning strategy: split_rows" in output
    assert "Detection OCR execution time:" in output
    assert (tmp_path / "detection_1_crop.jpg").is_file()
    assert (tmp_path / "detection_1_padding10_otsu.jpg").is_file()


def test_candidate_ranking_uses_validation_confidence_and_structure():
    invalid_high_confidence = Candidate(
        raw_text="292160344",
        normalized_text="292160344",
        ocr_confidence=0.99,
        preprocessing_variant="padding05_contrast_enhanced",
        is_valid=False,
        structure_score=plate_structure_score("292160344"),
        strategy="full",
        padding_ratio=0.05,
    )
    plausible = Candidate(
        raw_text="29F112345",
        normalized_text="29F112345",
        ocr_confidence=0.82,
        preprocessing_variant="padding10_otsu",
        is_valid=True,
        structure_score=plate_structure_score("29F112345"),
        strategy="full",
        padding_ratio=0.10,
    )
    less_structured = Candidate(
        raw_text="ABC12345",
        normalized_text="ABC12345",
        ocr_confidence=0.83,
        preprocessing_variant="padding10_grayscale",
        is_valid=True,
        structure_score=plate_structure_score("ABC12345"),
        strategy="full",
        padding_ratio=0.10,
    )
    assert select_best_candidate([invalid_high_confidence, plausible]) == plausible
    assert select_best_candidate([less_structured, plausible]) == plausible
    all_digit = validate_plate_candidate("292160344")
    assert all_digit.normalized_text == "292160344"
    assert all_digit.is_valid is False


def test_ocr_debug_fragments_and_balanced_beam_search(monkeypatch):
    class FakeReader:
        def readtext(self, _image, **kwargs):
            assert kwargs["decoder"] == "beamsearch"
            assert kwargs["beamWidth"] == 3
            return [
                ([[0, 0], [20, 0], [20, 10], [0, 10]], "29F1", 0.8),
                ([[22, 0], [42, 0], [42, 10], [22, 10]], "12345", 0.9),
            ]

    monkeypatch.setattr(ocr_module, "get_ocr_reader", lambda: FakeReader())
    result = run_ocr(np.zeros((20, 50), dtype=np.uint8), source_region="top")
    assert result.raw_text == "29F112345"
    assert len(result.fragments) == 2
    assert result.fragments[0].source_region == "top"
    assert result.fragments[0].center == (10.0, 5.0)


def teardown_module():
    from app.database import engine

    engine.dispose()
    if TEST_DB.exists():
        TEST_DB.unlink()
