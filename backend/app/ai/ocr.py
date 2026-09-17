import statistics
import threading
from dataclasses import dataclass

import numpy as np

from app.ai.model_manager import get_ocr_reader


ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ocr_inference_lock = threading.Lock()


@dataclass(frozen=True)
class OCRResult:
    raw_text: str
    confidence: float


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


def run_ocr(image: np.ndarray) -> OCRResult:
    reader = get_ocr_reader()
    with _ocr_inference_lock:
        fragments = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=ALLOWLIST,
        )
    if not fragments:
        return OCRResult(raw_text="", confidence=0.0)

    ordered = sort_fragments_spatially(list(fragments))
    raw_text = "".join(str(fragment[1]).strip() for fragment in ordered)
    weighted_total = 0.0
    total_length = 0
    for _, text, confidence in ordered:
        text_length = len(str(text).strip())
        if text_length:
            weighted_total += float(confidence) * text_length
            total_length += text_length
    aggregate = weighted_total / total_length if total_length else 0.0
    return OCRResult(raw_text=raw_text, confidence=max(0.0, min(1.0, aggregate)))
