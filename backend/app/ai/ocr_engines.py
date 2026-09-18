"""Optional OCR engine adapter used for benchmarks and controlled experiments.

Production continues to use EasyOCR through ``app.ai.ocr.run_ocr``. PaddleOCR is
loaded lazily only when a caller explicitly requests it and installs the optional
requirements; it is never run alongside EasyOCR during normal API recognition.
"""

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Callable

import numpy as np

from app.ai.ocr import run_ocr
from app.ai.validator import normalize_ocr_text, validate_plate_candidate


PRIMARY_OCR_ENGINE = os.getenv("PRIMARY_OCR_ENGINE", "easyocr").lower()
ENABLE_SECONDARY_OCR = os.getenv("ENABLE_SECONDARY_OCR", "false").lower() in {
    "1",
    "true",
    "yes",
}


@dataclass(frozen=True)
class EngineResult:
    engine: str
    raw_text: str
    normalized_text: str
    confidence: float
    is_valid: bool


class OCREngineUnavailable(RuntimeError):
    pass


def _engine_result(engine: str, raw_text: str, confidence: float) -> EngineResult:
    validation = validate_plate_candidate(raw_text)
    return EngineResult(
        engine=engine,
        raw_text=raw_text,
        normalized_text=validation.normalized_text,
        confidence=max(0.0, min(1.0, float(confidence))),
        is_valid=validation.is_valid,
    )


def recognize_with_easyocr(image: np.ndarray) -> EngineResult:
    result = run_ocr(image)
    return _engine_result("easyocr", result.raw_text, result.confidence)


@lru_cache(maxsize=1)
def _get_paddle_reader():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCREngineUnavailable(
            "PaddleOCR is not installed; install backend/requirements-paddle.txt"
        ) from exc
    try:
        return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except TypeError:
        return PaddleOCR(lang="en")


def _collect_paddle_text(value, texts: list[str], scores: list[float]) -> None:
    if isinstance(value, dict):
        if "rec_texts" in value:
            texts.extend(str(item) for item in value.get("rec_texts", []))
            scores.extend(float(item) for item in value.get("rec_scores", []))
            return
        for nested in value.values():
            _collect_paddle_text(nested, texts, scores)
        return
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], (int, float))
        ):
            texts.append(value[0])
            scores.append(float(value[1]))
            return
        for nested in value:
            _collect_paddle_text(nested, texts, scores)


def recognize_with_paddleocr(image: np.ndarray) -> EngineResult:
    reader = _get_paddle_reader()
    try:
        raw = reader.ocr(image, cls=True)
    except TypeError:
        raw = reader.predict(image)
    texts: list[str] = []
    scores: list[float] = []
    _collect_paddle_text(raw, texts, scores)
    normalized_parts = [normalize_ocr_text(text) for text in texts]
    raw_text = "".join(part for part in normalized_parts if part)
    confidence = sum(scores) / len(scores) if scores else 0.0
    return _engine_result("paddleocr", raw_text, confidence)


def recognize_with_optional_fallback(
    image: np.ndarray,
    *,
    primary: Callable[[np.ndarray], EngineResult] = recognize_with_easyocr,
    secondary: Callable[[np.ndarray], EngineResult] = recognize_with_paddleocr,
    enable_secondary: bool | None = None,
    minimum_confidence: float = 0.45,
) -> EngineResult:
    if enable_secondary is None:
        enable_secondary = ENABLE_SECONDARY_OCR
    first = primary(image)
    if not enable_secondary or (first.is_valid and first.confidence >= minimum_confidence):
        return first
    second = secondary(image)
    return max(
        (first, second),
        key=lambda result: (result.is_valid, result.confidence),
    )
