"""OCR engine adapters for Paddle-primary production recognition.

PaddleOCR stays fully lazy so PyTorch/YOLO can initialize its Windows native
runtime first. EasyOCR remains available as a conditional production fallback.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import json
from numbers import Real
import os
import threading
from typing import Callable

import numpy as np

from app.ai.ocr import run_ocr
from app.ai.validator import (
    normalize_ocr_text,
    plate_structure_score,
    validate_plate_candidate,
)


PRIMARY_OCR_ENGINE = os.getenv("PRIMARY_OCR_ENGINE", "paddleocr").lower()
ENABLE_SECONDARY_OCR = os.getenv("ENABLE_SECONDARY_OCR", "true").lower() in {
    "1",
    "true",
    "yes",
}
PADDLE_DEVICE = "cpu"
PADDLE_MKLDNN_ENABLED = False
_paddle_inference_lock = threading.Lock()


@dataclass(frozen=True)
class EngineFragment:
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...] = ()
    center: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class EngineResult:
    engine: str
    raw_text: str
    normalized_text: str
    confidence: float
    is_valid: bool
    structure_score: float = 0.0
    fragments: tuple[EngineFragment, ...] = ()
    noise_detected: bool = False


class OCREngineUnavailable(RuntimeError):
    pass


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def paddle_runtime_diagnostics() -> dict[str, str]:
    """Return opt-in benchmark diagnostics without logging during production OCR."""
    try:
        import paddle

        cuda_compiled = paddle.device.is_compiled_with_cuda()
    except ImportError:
        cuda_compiled = False
    return {
        "PaddlePaddle": _package_version("paddlepaddle"),
        "PaddleOCR": _package_version("paddleocr"),
        "PaddleX": _package_version("paddlex"),
        "CUDA compiled": "yes" if cuda_compiled else "no",
        "Device": PADDLE_DEVICE.upper(),
        "MKLDNN": "enabled" if PADDLE_MKLDNN_ENABLED else "disabled for PaddleOCR benchmark",
    }


def _engine_result(
    engine: str,
    raw_text: str,
    confidence: float,
    *,
    fragments: tuple[EngineFragment, ...] = (),
    noise_detected: bool = False,
) -> EngineResult:
    validation = validate_plate_candidate(raw_text)
    return EngineResult(
        engine=engine,
        raw_text=raw_text,
        normalized_text=validation.normalized_text,
        confidence=max(0.0, min(1.0, float(confidence))),
        is_valid=validation.is_valid,
        structure_score=validation.structure_score,
        fragments=fragments,
        noise_detected=noise_detected,
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
        return PaddleOCR(
            lang="en",
            device=PADDLE_DEVICE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=PADDLE_MKLDNN_ENABLED,
        )
    except (TypeError, ValueError) as exc:
        message = str(exc).lower()
        unsupported_marker = any(
            marker in message
            for marker in (
                "unexpected keyword argument",
                "unexpected argument",
                "unknown argument",
                "unsupported argument",
                "unrecognized argument",
            )
        )
        paddle_3_argument = any(
            argument in message
            for argument in (
                "device",
                "use_doc_orientation_classify",
                "use_doc_unwarping",
                "use_textline_orientation",
                "enable_mkldnn",
            )
        )
        unsupported_argument = unsupported_marker and paddle_3_argument
        if not unsupported_argument:
            raise
        return PaddleOCR(lang="en")


def initialize_paddleocr():
    """Initialize Paddle only after the caller has loaded the YOLO runtime."""
    return _get_paddle_reader()


def _geometry_points(value) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ()
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return ()
    if array.ndim == 1 and array.size == 4:
        x1, y1, x2, y2 = array.tolist()
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    if array.ndim == 2 and array.shape[1] >= 2:
        return tuple((float(point[0]), float(point[1])) for point in array)
    return ()


def _make_engine_fragment(text: str, confidence: float, geometry=None) -> EngineFragment:
    bbox = _geometry_points(geometry)
    if bbox:
        center = (
            sum(point[0] for point in bbox) / len(bbox),
            sum(point[1] for point in bbox) / len(bbox),
        )
    else:
        center = (0.0, 0.0)
    return EngineFragment(
        text=str(text),
        confidence=max(0.0, min(1.0, float(confidence))),
        bbox=bbox,
        center=center,
    )


def _sequence(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _collect_paddle_fragments(value, fragments: list[EngineFragment]) -> None:
    if value is None:
        return
    if not isinstance(value, (dict, list, tuple, str, bytes)) and hasattr(
        value, "json"
    ):
        payload = value.json
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        _collect_paddle_fragments(payload, fragments)
        return
    if isinstance(value, dict):
        if "rec_texts" in value:
            recognized_texts = _sequence(value.get("rec_texts"))
            recognized_scores = _sequence(value.get("rec_scores"))
            geometries = []
            for geometry_key in ("rec_polys", "rec_boxes", "dt_polys"):
                if value.get(geometry_key) is not None:
                    geometries = _sequence(value[geometry_key])
                    break
            for index, text in enumerate(recognized_texts):
                score = recognized_scores[index] if index < len(recognized_scores) else 0.0
                geometry = geometries[index] if index < len(geometries) else None
                fragments.append(_make_engine_fragment(str(text), float(score), geometry))
            return
        for nested in value.values():
            _collect_paddle_fragments(nested, fragments)
        return
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and isinstance(value[1], (list, tuple))
            and len(value[1]) == 2
            and isinstance(value[1][0], str)
            and isinstance(value[1][1], Real)
        ):
            fragments.append(
                _make_engine_fragment(value[1][0], float(value[1][1]), value[0])
            )
            return
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], Real)
        ):
            fragments.append(_make_engine_fragment(value[0], float(value[1])))
            return
        for nested in value:
            _collect_paddle_fragments(nested, fragments)


def _fragment_center_quality(fragment: EngineFragment, image_shape) -> float:
    if not fragment.bbox or image_shape is None:
        return 0.5
    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        return 0.5
    x = fragment.center[0] / width
    y = fragment.center[1] / height
    return 1.0 if 0.08 <= x <= 0.92 and 0.08 <= y <= 0.92 else 0.0


def _select_paddle_fragments(
    fragments: list[EngineFragment], image_shape
) -> tuple[str, float, tuple[EngineFragment, ...], bool]:
    usable = [fragment for fragment in fragments if normalize_ocr_text(fragment.text)]
    if not usable:
        return "", 0.0, (), False

    groups: list[tuple[int, ...]] = [(index,) for index in range(len(usable))]
    maximum_group = min(4, len(usable))
    for size in range(2, maximum_group + 1):
        for start in range(0, len(usable) - size + 1):
            groups.append(tuple(range(start, start + size)))

    ranked = []
    for indices in groups:
        selected = tuple(usable[index] for index in indices)
        raw_text = "".join(fragment.text for fragment in selected)
        normalized = normalize_ocr_text(raw_text)
        validation = validate_plate_candidate(normalized)
        confidence = sum(fragment.confidence for fragment in selected) / len(selected)
        length_quality = 1.0 if 7 <= len(normalized) <= 10 else 0.35 if 3 <= len(normalized) <= 12 else 0.0
        composition_quality = 1.0 if any(c.isalpha() for c in normalized) and any(c.isdigit() for c in normalized) else 0.0
        center_quality = sum(
            _fragment_center_quality(fragment, image_shape) for fragment in selected
        ) / len(selected)
        score = (
            0.38 * validation.structure_score
            + 0.22 * float(validation.is_valid)
            + 0.18 * confidence
            + 0.12 * length_quality
            + 0.06 * composition_quality
            + 0.04 * center_quality
        )
        ranked.append((score, validation.is_valid, confidence, -len(indices), raw_text, selected))

    _, _, confidence, _, raw_text, selected = max(
        ranked, key=lambda item: item[:5]
    )
    selected_ids = {id(fragment) for fragment in selected}
    noise_detected = any(
        id(fragment) not in selected_ids and len(normalize_ocr_text(fragment.text)) >= 2
        for fragment in usable
    )
    return raw_text, confidence, selected, noise_detected


def recognize_with_paddleocr(image: np.ndarray) -> EngineResult:
    reader = _get_paddle_reader()
    predict = getattr(reader, "predict", None)
    with _paddle_inference_lock:
        if callable(predict):
            raw = reader.predict(image)
        else:
            legacy_ocr = getattr(reader, "ocr", None)
            if not callable(legacy_ocr):
                raise OCREngineUnavailable(
                    "Installed PaddleOCR reader exposes neither predict() nor ocr()"
                )
            raw = legacy_ocr(image)
    fragments: list[EngineFragment] = []
    _collect_paddle_fragments(raw, fragments)
    raw_text, confidence, selected, noise_detected = _select_paddle_fragments(
        fragments, image.shape
    )
    return _engine_result(
        "paddleocr",
        raw_text,
        confidence,
        fragments=selected,
        noise_detected=noise_detected,
    )


def recognize_with_optional_fallback(
    image: np.ndarray,
    *,
    primary: Callable[[np.ndarray], EngineResult] = recognize_with_paddleocr,
    secondary: Callable[[np.ndarray], EngineResult] = recognize_with_easyocr,
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
