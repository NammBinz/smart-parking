import statistics
import threading
from dataclasses import dataclass

import numpy as np

from app.ai.model_manager import get_ocr_reader


ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ocr_inference_lock = threading.Lock()


@dataclass(frozen=True)
class OCRFragment:
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    source_region: str


@dataclass(frozen=True)
class OCRResult:
    raw_text: str
    confidence: float
    fragments: tuple[OCRFragment, ...] = ()


def _fragment_geometry(fragment) -> tuple[float, float, float]:
    points = fragment[0]
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    center_x = (min(x_values) + max(x_values)) / 2
    center_y = (min(y_values) + max(y_values)) / 2
    height = max(1.0, max(y_values) - min(y_values))
    return center_x, center_y, height


def sort_fragments_spatially(fragments: list) -> list:
    if len(fragments) < 2:
        return fragments

    geometries = [_fragment_geometry(fragment) for fragment in fragments]
    line_tolerance = max(5.0, statistics.median(item[2] for item in geometries) * 0.6)
    indexed = sorted(zip(fragments, geometries), key=lambda item: item[1][1])
    lines: list[list[tuple[object, tuple[float, float, float]]]] = []

    for fragment, geometry in indexed:
        best_line = None
        best_distance = None
        for line in lines:
            mean_y = sum(item[1][1] for item in line) / len(line)
            distance = abs(geometry[1] - mean_y)
            if distance <= line_tolerance and (best_distance is None or distance < best_distance):
                best_line = line
                best_distance = distance
        if best_line is None:
            lines.append([(fragment, geometry)])
        else:
            best_line.append((fragment, geometry))

    lines.sort(key=lambda line: sum(item[1][1] for item in line) / len(line))
    ordered = []
    for line in lines:
        line.sort(key=lambda item: item[1][0])
        ordered.extend(item[0] for item in line)
    return ordered


def run_ocr(image: np.ndarray, source_region: str = "full") -> OCRResult:
    reader = get_ocr_reader()
    with _ocr_inference_lock:
        fragments = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=ALLOWLIST,
            decoder="beamsearch",
            beamWidth=3,
        )
    if not fragments:
        return OCRResult(raw_text="", confidence=0.0)

    ordered = sort_fragments_spatially(list(fragments))
    raw_text = "".join(str(fragment[1]).strip() for fragment in ordered)
    weighted_total = 0.0
    total_length = 0
    debug_fragments = []
    for bbox, text, confidence in ordered:
        text_length = len(str(text).strip())
        if text_length:
            weighted_total += float(confidence) * text_length
            total_length += text_length
        points = tuple((float(point[0]), float(point[1])) for point in bbox)
        center_x, center_y, _ = _fragment_geometry((bbox, text, confidence))
        debug_fragments.append(
            OCRFragment(
                text=str(text),
                confidence=max(0.0, min(1.0, float(confidence))),
                bbox=points,
                center=(center_x, center_y),
                source_region=source_region,
            )
        )
    aggregate = weighted_total / total_length if total_length else 0.0
    return OCRResult(
        raw_text=raw_text,
        confidence=max(0.0, min(1.0, aggregate)),
        fragments=tuple(debug_fragments),
    )


def combine_ocr_results(top: OCRResult, bottom: OCRResult) -> OCRResult:
    raw_text = f"{top.raw_text}{bottom.raw_text}"
    top_length = len(top.raw_text.strip())
    bottom_length = len(bottom.raw_text.strip())
    total_length = top_length + bottom_length
    confidence = (
        ((top.confidence * top_length) + (bottom.confidence * bottom_length)) / total_length
        if total_length
        else 0.0
    )
    return OCRResult(
        raw_text=raw_text,
        confidence=max(0.0, min(1.0, confidence)),
        fragments=top.fragments + bottom.fragments,
    )
