import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.ai.model_manager import BACKEND_DIR, ModelInitializationError, runtime_status
from app.ai.pipeline import recognize_license_plates
from app.database import get_db
from app.models import Setting
from app.schemas.ai import AIStatusResponse, FrameRecognitionResponse, RecognitionResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])
UPLOAD_DIR = BACKEND_DIR / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}


def _image_extension(contents: bytes) -> str | None:
    if contents.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return None


async def _decode_image_upload(file: UploadFile | None) -> tuple[bytes, str, np.ndarray]:
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is required")
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are supported",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file exceeds the 10 MB limit",
        )

    extension = _image_extension(contents)
    if extension is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image data")
    encoded = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image could not be decoded")
    return contents, extension, image


async def _recognize(image: np.ndarray, confidence: float) -> list[dict[str, object]]:
    try:
        return await run_in_threadpool(recognize_license_plates, image, confidence)
    except ModelInitializationError as exc:
        logger.exception("AI runtime initialization failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("License plate recognition failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="License plate recognition failed",
        ) from exc


@router.get("/status", response_model=AIStatusResponse)
def get_ai_status():
    try:
        return runtime_status()
    except ModelInitializationError as exc:
        logger.exception("AI model status check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post("/recognize-image", response_model=RecognitionResponse)
async def recognize_image(
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    contents, extension, image = await _decode_image_upload(file)

    settings = db.scalar(select(Setting).limit(1))
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Settings not configured",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    generated_name = f"{uuid4()}{extension}"
    stored_path = UPLOAD_DIR / generated_name
    try:
        stored_path.write_bytes(contents)
    except OSError as exc:
        logger.exception("Could not store uploaded recognition image")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image could not be stored",
        ) from exc
    detections = await _recognize(image, float(settings.yolo_confidence))

    image_height, image_width = image.shape[:2]
    recognized = bool(detections)
    return RecognitionResponse(
        recognized=recognized,
        image_path=f"/uploads/{generated_name}",
        image_width=image_width,
        image_height=image_height,
        detections=detections,
        best_index=0 if detections else None,
        message=None if recognized else "No license plate detected",
    )


@router.post("/recognize-frame", response_model=FrameRecognitionResponse)
async def recognize_frame(
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    _, _, image = await _decode_image_upload(file)
    settings = db.scalar(select(Setting).limit(1))
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Settings not configured",
        )

    detections = await _recognize(image, float(settings.yolo_confidence))
    image_height, image_width = image.shape[:2]
    recognized = bool(detections)
    return FrameRecognitionResponse(
        recognized=recognized,
        image_width=image_width,
        image_height=image_height,
        detections=detections,
        best_index=0 if detections else None,
        message=None if recognized else "No license plate detected",
    )
