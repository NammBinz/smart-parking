from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from app.ai.detector import Detection, detect_plates
from app.ai.ocr import OCRFragment, OCRResult, combine_ocr_results, run_ocr
from app.ai.preprocess import (
    crop_to_bbox,
    crop_with_padding,
    preprocessing_variants,
    split_two_line_crop,
)
from app.ai.validator import validate_plate_candidate


PADDING_OPTIONS = ((0.05, "padding05"), (0.10, "padding10"), (0.15, "padding15"))
ALL_VARIANTS = (
    "upscaled_color",
    "grayscale",
    "contrast_enhanced",
    "adaptive_threshold",
    "otsu",
    "bilateral_otsu",
    "sharpened_grayscale",
)

# Normal inference exercises every preprocessing method at 10% padding while
# retaining the strongest existing contrast candidate at 5% and 15%.
NORMAL_FULL_PLAN = {
    "padding05": ("contrast_enhanced",),
    "padding10": ALL_VARIANTS,
    "padding15": ("contrast_enhanced",),
}
NORMAL_SPLIT_PLAN = {"padding10": ("contrast_enhanced", "otsu")}


@dataclass(frozen=True)
class Candidate:
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    preprocessing_variant: str
    is_valid: bool
    structure_score: float
    strategy: str
    padding_ratio: float
    fragments: tuple[OCRFragment, ...] = ()
    execution_seconds: float = 0.0


@dataclass
class DetectionAnalysis:
    detection: Detection
    winner: Candidate
    candidates: list[Candidate]
    split_row_used: bool
    ocr_execution_seconds: float
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    def public_result(self) -> dict[str, object]:
        x1, y1, x2, y2 = self.detection.bbox
        return {
            "class_id": self.detection.class_id,
            "class_name": self.detection.class_name,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "detection_confidence": self.detection.confidence,
            "raw_text": self.winner.raw_text,
            "normalized_text": self.winner.normalized_text,
            "ocr_confidence": self.winner.ocr_confidence,
            "preprocessing_variant": self.winner.preprocessing_variant,
            "is_valid": self.winner.is_valid,
        }


def _candidate_from_result(
    result: OCRResult,
    identity: str,
    strategy: str,
    padding_ratio: float,
    execution_seconds: float,
) -> Candidate:
    validation = validate_plate_candidate(result.raw_text)
    return Candidate(
        raw_text=result.raw_text,
        normalized_text=validation.normalized_text,
        ocr_confidence=result.confidence,
        preprocessing_variant=identity,
        is_valid=validation.is_valid,
        structure_score=validation.structure_score,
        strategy=strategy,
        padding_ratio=padding_ratio,
        fragments=result.fragments,
        execution_seconds=execution_seconds,
    )


def candidate_ranking_key(candidate: Candidate) -> tuple[bool, float, float, float]:
    # Validation remains dominant. Structure contributes only a small bonus,
    # allowing it to separate candidates whose OCR confidence is close.
    quality = candidate.ocr_confidence + (0.08 * candidate.structure_score)
    return (
        candidate.is_valid,
        quality,
        candidate.ocr_confidence,
        candidate.structure_score,
    )


def select_best_candidate(candidates: list[Candidate]) -> Candidate:
    return max(candidates, key=candidate_ranking_key, default=_empty_candidate())


def _empty_candidate() -> Candidate:
    return Candidate(
        raw_text="",
        normalized_text="",
        ocr_confidence=0.0,
        preprocessing_variant="none",
        is_valid=False,
        structure_score=0.0,
        strategy="full",
        padding_ratio=0.0,
    )


def _timed_ocr(image: np.ndarray, source_region: str) -> tuple[OCRResult, float]:
    started = perf_counter()
    result = run_ocr(image, source_region=source_region)
    return result, perf_counter() - started


def _selected_variants(
    padding_label: str,
    exhaustive: bool,
    split_rows: bool,
) -> tuple[str, ...]:
    if exhaustive:
        return ALL_VARIANTS
    plan = NORMAL_SPLIT_PLAN if split_rows else NORMAL_FULL_PLAN
    return plan.get(padding_label, ())


def _analyze_detection(
    image: np.ndarray,
    detection: Detection,
    exhaustive: bool,
    capture_images: bool,
) -> DetectionAnalysis:
    candidates: list[Candidate] = []
    debug_images: dict[str, np.ndarray] = {}
    raw_crop = crop_to_bbox(image, detection.bbox)
    if capture_images:
        debug_images["crop"] = raw_crop

    raw_height, raw_width = raw_crop.shape[:2]
    likely_two_line = raw_width > 0 and (raw_height / raw_width) >= 0.65

    for padding_ratio, padding_label in PADDING_OPTIONS:
        padded_crop = crop_with_padding(image, detection.bbox, padding_ratio)
        variants = preprocessing_variants(padded_crop)
        if capture_images:
            debug_images[f"{padding_label}_crop"] = padded_crop
            for variant_name, variant_image in variants.items():
                debug_images[f"{padding_label}_{variant_name}"] = variant_image

        for variant_name in _selected_variants(padding_label, exhaustive, split_rows=False):
            result, elapsed = _timed_ocr(variants[variant_name], source_region="full")
            candidates.append(
                _candidate_from_result(
                    result,
                    identity=f"{padding_label}_{variant_name}",
                    strategy="full",
                    padding_ratio=padding_ratio,
                    execution_seconds=elapsed,
                )
            )

        split_variants = _selected_variants(padding_label, exhaustive, split_rows=True)
        if likely_two_line and split_variants:
            top_crop, bottom_crop = split_two_line_crop(padded_crop)
            top_variants = preprocessing_variants(top_crop)
            bottom_variants = preprocessing_variants(bottom_crop)
            if capture_images:
                debug_images[f"{padding_label}_split_top_crop"] = top_crop
                debug_images[f"{padding_label}_split_bottom_crop"] = bottom_crop
                for variant_name in split_variants:
                    debug_images[f"{padding_label}_split_top_{variant_name}"] = top_variants[
                        variant_name
                    ]
                    debug_images[f"{padding_label}_split_bottom_{variant_name}"] = bottom_variants[
                        variant_name
                    ]

            for variant_name in split_variants:
                top_result, top_elapsed = _timed_ocr(
                    top_variants[variant_name],
                    source_region="top",
                )
                bottom_result, bottom_elapsed = _timed_ocr(
                    bottom_variants[variant_name],
                    source_region="bottom",
                )
                combined = combine_ocr_results(top_result, bottom_result)
                candidates.append(
                    _candidate_from_result(
                        combined,
                        identity=f"{padding_label}_split_rows_{variant_name}",
                        strategy="split_rows",
                        padding_ratio=padding_ratio,
                        execution_seconds=top_elapsed + bottom_elapsed,
                    )
                )

    winner = select_best_candidate(candidates)
    return DetectionAnalysis(
        detection=detection,
        winner=winner,
        candidates=candidates,
        split_row_used=likely_two_line,
        ocr_execution_seconds=sum(candidate.execution_seconds for candidate in candidates),
        debug_images=debug_images,
    )


def _analysis_ranking_key(analysis: DetectionAnalysis) -> tuple[bool, float]:
    combined = (
        0.55 * analysis.winner.ocr_confidence
        + 0.35 * analysis.detection.confidence
        + 0.10 * analysis.winner.structure_score
    )
    return analysis.winner.is_valid, combined


def analyze_license_plates(
    image: np.ndarray,
    confidence_threshold: float,
    *,
    exhaustive: bool = False,
    capture_images: bool = False,
) -> list[DetectionAnalysis]:
    detections = detect_plates(image, confidence_threshold)
    analyses = [
        _analyze_detection(image, detection, exhaustive, capture_images)
        for detection in detections
    ]
    analyses.sort(key=_analysis_ranking_key, reverse=True)
    return analyses


def recognize_license_plates(image: np.ndarray, confidence_threshold: float) -> list[dict[str, object]]:
    return [
        analysis.public_result()
        for analysis in analyze_license_plates(image, confidence_threshold)
    ]
