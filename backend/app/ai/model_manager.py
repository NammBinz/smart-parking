import importlib.util
import threading
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BACKEND_DIR / "models" / "best.pt"
PUBLIC_MODEL_PATH = "backend/models/best.pt"

_yolo_model: Any | None = None
_ocr_reader: Any | None = None
_yolo_lock = threading.Lock()
_ocr_lock = threading.Lock()


class ModelInitializationError(RuntimeError):
    """Raised when an AI runtime component cannot be initialized."""


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        with _yolo_lock:
            if _yolo_model is None:
                if not MODEL_PATH.is_file():
                    raise ModelInitializationError("License plate model file is unavailable")
                try:
                    from ultralytics import YOLO

                    _yolo_model = YOLO(str(MODEL_PATH))
                except Exception as exc:
                    raise ModelInitializationError("License plate model could not be loaded") from exc
    return _yolo_model


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                try:
                    import easyocr
                    import torch

                    _ocr_reader = easyocr.Reader(["en"], gpu=bool(torch.cuda.is_available()))
                except Exception as exc:
                    raise ModelInitializationError("OCR engine could not be initialized") from exc
    return _ocr_reader


def model_names() -> dict[str, str]:
    names = get_yolo_model().names
    if isinstance(names, dict):
        return {str(key): str(value) for key, value in names.items()}
    return {str(index): str(value) for index, value in enumerate(names)}


def runtime_status() -> dict[str, object]:
    yolo_installed = dependency_available("ultralytics")
    paddle_installed = dependency_available("paddleocr") and dependency_available(
        "paddle"
    )
    easyocr_installed = dependency_available("easyocr")
    available = MODEL_PATH.is_file() and yolo_installed
    names: dict[str, str] = {}
    if available:
        names = model_names()
    return {
        "model_available": available,
        "model_path": PUBLIC_MODEL_PATH,
        "model_names": names,
        "ocr_available": paddle_installed and easyocr_installed,
    }
