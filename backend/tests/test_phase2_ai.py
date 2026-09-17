from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.ai.ocr import sort_fragments_spatially
from app.ai.preprocess import crop_with_padding, preprocessing_variants
from app.ai.validator import validate_plate_candidate
from app.api import ai as ai_api
from app.database import SessionLocal
from app.main import app
from app.models import ParkingSession


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


def teardown_module():
    from app.database import engine

    engine.dispose()
    if TEST_DB.exists():
        TEST_DB.unlink()
