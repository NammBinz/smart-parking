from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api import ai as ai_api
from app.ai.detector import Detection
from app.ai import frame_quality as frame_quality_module
from app.ai.frame_quality import (
    FrameQuality,
    FrameQualityAnalysis,
    calculate_frame_quality_score,
    calculate_sharpness,
    analyze_frame_quality,
)
from app.main import app


TEST_DB = Path(__file__).parent / "test_parking.db"


def _jpeg_bytes() -> bytes:
    image = np.full((48, 96, 3), 170, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def _detection(text="29Z158344", *, status="ok", valid=True):
    return {
        "class_id": 1,
        "class_name": "plate_two_line",
        "bbox": {"x1": 10, "y1": 8, "x2": 80, "y2": 42},
        "detection_confidence": 0.94,
        "raw_text": text,
        "normalized_text": text,
        "ocr_confidence": 0.86 if valid else 0.0,
        "preprocessing_variant": "row_fusion" if valid else "none",
        "is_valid": valid,
        "ocr_status": status,
    }


def test_camera_frame_validation(monkeypatch):
    monkeypatch.setattr(ai_api, "recognize_license_plates", lambda *_args: [])
    with TestClient(app) as client:
        assert client.post("/api/ai/recognize-frame").status_code == 400
        unsupported = client.post(
            "/api/ai/recognize-frame",
            files={"file": ("frame.gif", b"GIF89a", "image/gif")},
        )
        assert unsupported.status_code == 400
        invalid = client.post(
            "/api/ai/recognize-frame",
            files={"file": ("frame.jpg", b"\xff\xd8\xffinvalid", "image/jpeg")},
        )
        assert invalid.status_code == 400


def test_camera_frames_are_memory_only_and_return_compatible_results(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_api, "UPLOAD_DIR", tmp_path)
    responses = [
        [_detection()],
        [],
        [_detection("", status="too_small", valid=False)],
    ]
    monkeypatch.setattr(
        ai_api,
        "recognize_license_plates",
        lambda _image, _threshold: responses.pop(0),
    )

    with TestClient(app) as client:
        success = client.post(
            "/api/ai/recognize-frame",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert success.status_code == 200
        assert success.json()["recognized"] is True
        assert success.json()["detections"][0]["normalized_text"] == "29Z158344"
        assert "image_path" not in success.json()

        no_plate = client.post(
            "/api/ai/recognize-frame",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert no_plate.status_code == 200
        assert no_plate.json()["recognized"] is False
        assert no_plate.json()["best_index"] is None

        too_small = client.post(
            "/api/ai/recognize-frame",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert too_small.status_code == 200
        assert too_small.json()["detections"][0]["ocr_status"] == "too_small"
        assert too_small.json()["detections"][0]["is_valid"] is False

    assert list(tmp_path.iterdir()) == []


def test_sharpness_and_quality_score_are_bounded():
    checkerboard = np.indices((80, 160)).sum(axis=0) % 2 * 255
    checkerboard = checkerboard.astype(np.uint8)
    blurred = cv2.GaussianBlur(checkerboard, (21, 21), 0)
    assert calculate_sharpness(checkerboard) > calculate_sharpness(blurred)
    assert abs(calculate_frame_quality_score(2, -1, 2, 1, 0.5) - 0.70) < 1e-9


def test_blurry_camera_candidate_waits_without_ocr(monkeypatch):
    detection = Detection(0, "plate", 0.9, (10, 10, 100, 45))
    monkeypatch.setattr(
        frame_quality_module, "detect_plates", lambda *_args: [detection]
    )
    image = np.full((80, 140, 3), 128, dtype=np.uint8)
    analysis = analyze_frame_quality(image, 0.5)
    assert analysis.detections[0].camera_status == "too_blurry"


def test_analyze_frame_returns_quality_without_storing_or_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_api, "UPLOAD_DIR", tmp_path)
    detection = Detection(0, "plate", 0.91, (10, 8, 80, 42))
    analysis = FrameQualityAnalysis(
        detections=(
            FrameQuality(
                detection=detection,
                width=70,
                height=34,
                area=2380,
                sharpness=125.0,
                brightness=130.0,
                brightness_quality=0.98,
                size_quality=0.62,
                center_bonus=1.0,
                quality_score=0.81,
                camera_status="candidate",
            ),
        ),
        yolo_seconds=0.012,
        quality_seconds=0.002,
    )
    monkeypatch.setattr(ai_api, "analyze_frame_quality", lambda *_args: analysis)
    monkeypatch.setattr(
        ai_api,
        "recognize_license_plates",
        lambda *_args: (_ for _ in ()).throw(AssertionError("OCR must not run")),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/analyze-frame",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["detections"][0]["camera_status"] == "candidate"
    assert payload["detections"][0]["sharpness"] == 125.0
    assert payload["yolo_inference_time"] == 0.012
    assert list(tmp_path.iterdir()) == []


def test_upload_endpoint_still_persists_original(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(ai_api, "recognize_license_plates", lambda *_args: [_detection()])
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/recognize-image",
            files={"file": ("vehicle.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json()["image_path"].startswith("/uploads/")
    stored = list(tmp_path.iterdir())
    assert len(stored) == 1
    assert stored[0].suffix == ".jpg"


def teardown_module():
    from app.database import engine

    engine.dispose()
    if TEST_DB.exists():
        TEST_DB.unlink()
