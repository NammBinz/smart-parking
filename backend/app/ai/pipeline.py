from collections import defaultdict
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Callable

import numpy as np

from app.ai.detector import Detection, detect_plates
from app.ai.ocr import OCRFragment, OCRResult, combine_ocr_results, run_ocr
from app.ai.preprocess import (
    PreprocessingCache,
    crop_to_bbox,
    crop_with_padding,
    preprocessing_variants,
    split_two_line_crop,
    tight_top_row_crop,
)
from app.ai.quality import assess_detection_quality
from app.ai.validator import normalize_ocr_text, validate_plate_candidate


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
ONE_LINE_NORMAL_PLAN = {
    "padding10": ("grayscale", "otsu", "sharpened_grayscale"),
}
MAX_NORMAL_OCR_CALLS = 6
STRONG_TOP_SCORE = 0.72
STRONG_BOTTOM_SCORE = 0.70
TOP_CONFUSIONS = {"5": ("S",), "2": ("Z",), "8": ("B",), "0": ("O",), "1": ("I", "L")}
BOTTOM_CONFUSIONS = {
    "S": ("5",),
    "Z": ("2",),
    "B": ("8",),
    "O": ("0",),
    "I": ("1",),
    "L": ("1", "4"),
}


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
    evidence_score: float = 0.0
    correction_used: bool = False
    correction_count: int = 0


@dataclass(frozen=True)
class RowCandidate:
    row: str
    raw_text: str
    normalized_text: str
    confidence: float
    source: str
    preprocessing_variant: str
    padding_label: str
    decoder: str = "greedy"
    correction_count: int = 0
    original_text: str = ""


@dataclass(frozen=True)
class RowSelection:
    row: str
    text: str
    confidence: float
    score: float
    consensus_count: int
    diversity_count: int
    sources: tuple[str, ...]
    raw_text: str = ""
    correction_used: bool = False
    correction_count: int = 0


@dataclass
class _TimingMetrics:
    preprocessing_seconds: float = 0.0
    ocr_execution_seconds: float = 0.0
    ranking_seconds: float = 0.0
    ocr_calls: int = 0
    decoders: list[str] = field(default_factory=list)


@dataclass
class DetectionAnalysis:
    detection: Detection
    winner: Candidate
    candidates: list[Candidate]
    split_row_used: bool
    row_fusion_used: bool
    top_row: RowSelection | None
    bottom_row: RowSelection | None
    row_candidates: list[RowCandidate]
    ocr_execution_seconds: float
    ocr_status: str = "ok"
    correction_used: bool = False
    correction_count: int = 0
    ocr_calls: int = 0
    decoders_used: tuple[str, ...] = ()
    yolo_inference_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    ranking_seconds: float = 0.0
    total_recognition_seconds: float = 0.0
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def top_row_text(self) -> str:
        return self.top_row.text if self.top_row else ""

    @property
    def top_row_confidence(self) -> float:
        return self.top_row.confidence if self.top_row else 0.0

    @property
    def bottom_row_text(self) -> str:
        return self.bottom_row.text if self.bottom_row else ""

    @property
    def bottom_row_confidence(self) -> float:
        return self.bottom_row.confidence if self.bottom_row else 0.0

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
            "ocr_status": self.ocr_status,
        }


ProgressCallback = Callable[[int, int, str, float], None]


@dataclass
class _ProgressTracker:
    total: int
    callback: ProgressCallback | None
    completed: int = 0
    started: float = field(default_factory=perf_counter)

    def update(self, label: str) -> None:
        self.completed += 1
        if self.callback:
            self.callback(self.completed, self.total, label, perf_counter() - self.started)


def _empty_candidate() -> Candidate:
    return Candidate("", "", 0.0, "none", False, 0.0, "full", 0.0)


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
    quality = candidate.ocr_confidence + (0.08 * candidate.structure_score)
    return candidate.is_valid, quality, candidate.ocr_confidence, candidate.structure_score


def select_best_candidate(candidates: list[Candidate]) -> Candidate:
    return max(candidates, key=candidate_ranking_key, default=_empty_candidate())


def top_row_structure_score(text: str) -> float:
    normalized = normalize_ocr_text(text)
    if not normalized or not normalized.isalnum():
        return 0.0
    letters = sum(character.isalpha() for character in normalized)
    score = 0.05
    if len(normalized) >= 2 and normalized[:2].isdigit():
        score += 0.25
    if len(normalized) >= 3 and normalized[2].isalpha():
        score += 0.35
    elif letters >= 1:
        score += 0.15
    if len(normalized) == 4:
        score += 0.20
    elif len(normalized) == 5:
        score += 0.10
    elif len(normalized) in {3, 6}:
        score += 0.05
    if 1 <= letters <= 2:
        score += 0.10
    return min(1.0, score)


def bottom_row_structure_score(text: str) -> float:
    normalized = normalize_ocr_text(text)
    if not normalized or not normalized.isalnum():
        return 0.0
    score = 0.05
    if normalized.isdigit():
        score += 0.55
    if len(normalized) in {4, 5}:
        score += 0.28
    elif len(normalized) == 6:
        score += 0.10
    if sum(character.isdigit() for character in normalized) >= 4:
        score += 0.10
    return min(1.0, score)


def _positional_similarity(left: str, right: str) -> float:
    width = max(len(left), len(right))
    if width == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(left, right))
    return matches / width


def _corrected_row_alternatives(candidate: RowCandidate) -> list[RowCandidate]:
    """Generate bounded, provenance-preserving alternatives for row-specific confusions."""
    text = candidate.normalized_text
    replacements: list[tuple[int, tuple[str, ...]]] = []
    if candidate.row == "top" and 3 <= len(text) <= 5 and text[:2].isdigit():
        if text[2] in TOP_CONFUSIONS:
            replacements.append((2, TOP_CONFUSIONS[text[2]]))
    elif candidate.row == "bottom" and 4 <= len(text) <= 6:
        for index, character in enumerate(text):
            if character in BOTTOM_CONFUSIONS:
                replacements.append((index, BOTTOM_CONFUSIONS[character]))

    alternatives: list[tuple[str, int]] = [(text, 0)]
    for index, values in replacements[:2]:
        additions = []
        for current, count in alternatives:
            for replacement in values:
                additions.append((current[:index] + replacement + current[index + 1 :], count + 1))
        alternatives.extend(additions)

    unique: dict[str, int] = {}
    for alternative, count in alternatives:
        if alternative != text:
            unique[alternative] = min(count, unique.get(alternative, count))
    return [
        replace(
            candidate,
            normalized_text=alternative,
            correction_count=count,
            original_text=text,
        )
        for alternative, count in list(unique.items())[:8]
    ]


def _composite_top_candidates(row_candidates: list[RowCandidate]) -> list[RowCandidate]:
    prefixes = [
        candidate
        for candidate in row_candidates
        if candidate.row == "top"
        and len(candidate.normalized_text) >= 3
        and candidate.normalized_text[:2].isdigit()
        and candidate.confidence >= 0.15
    ]
    suffixes = [
        candidate
        for candidate in row_candidates
        if candidate.row == "top"
        and len(candidate.normalized_text) == 2
        and candidate.normalized_text[0].isalpha()
        and candidate.normalized_text[1].isdigit()
        and candidate.confidence >= 0.15
    ]
    composites: list[RowCandidate] = []
    seen: set[tuple[str, str]] = set()
    for prefix in prefixes:
        for suffix in suffixes:
            if prefix.source == suffix.source:
                continue
            text = prefix.normalized_text[:2] + suffix.normalized_text
            source = f"composite:{prefix.source}+{suffix.source}"
            key = (text, source)
            if key in seen:
                continue
            seen.add(key)
            composites.append(
                RowCandidate(
                    row="top",
                    raw_text=f"{prefix.raw_text}|{suffix.raw_text}",
                    normalized_text=text,
                    confidence=(prefix.confidence + suffix.confidence) / 2,
                    source=source,
                    preprocessing_variant=f"{prefix.preprocessing_variant}+{suffix.preprocessing_variant}",
                    padding_label=f"{prefix.padding_label}+{suffix.padding_label}",
                    decoder=f"{prefix.decoder}+{suffix.decoder}",
                    original_text=text,
                )
            )
    return composites


def expand_row_candidates(row_candidates: list[RowCandidate]) -> list[RowCandidate]:
    originals = list(row_candidates) + _composite_top_candidates(row_candidates)
    expanded = list(originals)
    for candidate in originals:
        if candidate.correction_count == 0:
            expanded.extend(_corrected_row_alternatives(candidate))
    return expanded


def select_best_row_candidate(
    row_candidates: list[RowCandidate],
    row: str,
) -> RowSelection | None:
    unique_candidates: dict[tuple[str, str, int], RowCandidate] = {}
    for candidate in expand_row_candidates(row_candidates):
        if candidate.row == row and candidate.normalized_text:
            key = (candidate.normalized_text, candidate.source, candidate.correction_count)
            current = unique_candidates.get(key)
            if current is None or candidate.confidence > current.confidence:
                unique_candidates[key] = candidate
    candidates = list(unique_candidates.values())
    if not candidates:
        return None

    grouped: dict[str, list[RowCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.normalized_text].append(candidate)
    distinct_texts = list(grouped)
    structure_function = top_row_structure_score if row == "top" else bottom_row_structure_score
    selections = []

    for text, evidence in grouped.items():
        confidences = [candidate.confidence for candidate in evidence]
        confidence = (0.6 * max(confidences)) + (0.4 * (sum(confidences) / len(confidences)))
        consensus_count = len({candidate.source for candidate in evidence})
        diversity_count = len({candidate.preprocessing_variant for candidate in evidence})
        consensus_bonus = min(consensus_count, 3) / 3
        diversity_bonus = min(diversity_count, 3) / 3
        neighborhood = sum(
            _positional_similarity(text, other_text) for other_text in distinct_texts
        ) / len(distinct_texts)
        if row == "top":
            score = (
                0.60 * structure_function(text)
                + 0.18 * confidence
                + 0.12 * consensus_bonus
                + 0.05 * diversity_bonus
                + 0.05 * neighborhood
            )
        else:
            score = (
                0.45 * structure_function(text)
                + 0.40 * confidence
                + 0.08 * consensus_bonus
                + 0.04 * diversity_bonus
                + 0.03 * neighborhood
            )
        correction_count = min(candidate.correction_count for candidate in evidence)
        direct_evidence = any(candidate.correction_count == 0 for candidate in evidence)
        if correction_count:
            score -= 0.08 * correction_count
            # A table-only substitution is not enough. Require either a direct OCR
            # observation or the same correction implied by independent variants.
            if not direct_evidence and diversity_count < 2:
                score -= 0.30
            elif not direct_evidence:
                score -= 0.05
        representative = max(
            evidence,
            key=lambda candidate: (
                candidate.correction_count == 0,
                candidate.confidence,
                -candidate.correction_count,
            ),
        )
        selections.append(
            RowSelection(
                row=row,
                text=text,
                confidence=confidence,
                score=score,
                consensus_count=consensus_count,
                diversity_count=diversity_count,
                sources=tuple(sorted(candidate.source for candidate in evidence)),
                raw_text=representative.raw_text,
                correction_used=correction_count > 0,
                correction_count=correction_count,
            )
        )
    return max(selections, key=lambda selection: (selection.score, selection.confidence))


def _row_candidate(
    row: str,
    raw_text: str,
    confidence: float,
    source: str,
    variant: str,
    padding_label: str,
    decoder: str = "greedy",
) -> RowCandidate | None:
    normalized = normalize_ocr_text(raw_text)
    if not normalized:
        return None
    return RowCandidate(
        row=row,
        raw_text=raw_text,
        normalized_text=normalized,
        confidence=confidence,
        source=source,
        preprocessing_variant=variant,
        padding_label=padding_label,
        decoder=decoder,
        original_text=normalized,
    )


def _fragment_bounds(fragment: OCRFragment) -> tuple[float, float, float, float]:
    if not fragment.bbox:
        return fragment.center[0], fragment.center[1], fragment.center[0], fragment.center[1]
    xs = [point[0] for point in fragment.bbox]
    ys = [point[1] for point in fragment.bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_ratio(left: OCRFragment, right: OCRFragment) -> float:
    lx1, ly1, lx2, ly2 = _fragment_bounds(left)
    rx1, ry1, rx2, ry2 = _fragment_bounds(right)
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    smaller = min(max(1.0, (lx2 - lx1) * (ly2 - ly1)), max(1.0, (rx2 - rx1) * (ry2 - ry1)))
    return intersection / smaller


def _deduplicate_overlapping_fragments(fragments: list[OCRFragment]) -> list[OCRFragment]:
    accepted: list[OCRFragment] = []
    for fragment in sorted(fragments, key=lambda item: item.center[0]):
        duplicate_index = next(
            (index for index, current in enumerate(accepted) if _overlap_ratio(current, fragment) >= 0.45),
            None,
        )
        if duplicate_index is None:
            accepted.append(fragment)
            continue
        current = accepted[duplicate_index]
        current_text = normalize_ocr_text(current.text)
        fragment_text = normalize_ocr_text(fragment.text)
        current_value = (len(current_text), current.confidence)
        fragment_value = (len(fragment_text), fragment.confidence)
        if fragment_value > current_value:
            accepted[duplicate_index] = fragment
    return sorted(accepted, key=lambda item: item.center[0])


def _row_candidates_from_full_result(
    result: OCRResult,
    image_height: int,
    source: str,
    variant: str,
    padding_label: str,
) -> list[RowCandidate]:
    rows: dict[str, list[OCRFragment]] = {"top": [], "bottom": []}
    for fragment in result.fragments:
        row = "top" if fragment.center[1] < (image_height / 2) else "bottom"
        rows[row].append(fragment)

    candidates: list[RowCandidate] = []
    for row, fragments in rows.items():
        fragments.sort(key=lambda fragment: fragment.center[0])
        for index, fragment in enumerate(fragments):
            candidate = _row_candidate(
                row,
                fragment.text,
                fragment.confidence,
                f"{source}:fragment{index + 1}",
                variant,
                padding_label,
            )
            if candidate:
                candidates.append(candidate)
        merged_fragments = _deduplicate_overlapping_fragments(fragments)
        if len(merged_fragments) > 1:
            raw_text = "".join(fragment.text for fragment in merged_fragments)
            total_length = sum(len(fragment.text.strip()) for fragment in merged_fragments)
            confidence = (
                sum(
                    fragment.confidence * len(fragment.text.strip())
                    for fragment in merged_fragments
                )
                / total_length
                if total_length
                else 0.0
            )
            combined = _row_candidate(
                row,
                raw_text,
                confidence,
                f"{source}:combined_{row}",
                variant,
                padding_label,
            )
            if combined:
                candidates.append(combined)
    return candidates


def _timed_ocr(
    image: np.ndarray,
    source_region: str,
    decoder: str,
    label: str,
    tracker: _ProgressTracker,
    metrics: _TimingMetrics,
) -> tuple[OCRResult, float]:
    started = perf_counter()
    result = run_ocr(image, source_region=source_region, decoder=decoder)
    elapsed = perf_counter() - started
    metrics.ocr_execution_seconds += elapsed
    metrics.ocr_calls += 1
    metrics.decoders.append(decoder)
    tracker.update(label)
    return result, elapsed


def _fused_candidate(
    top: RowSelection | None,
    bottom: RowSelection | None,
) -> Candidate | None:
    if top is None or bottom is None:
        return None
    raw_text = f"{top.raw_text or top.text}{bottom.raw_text or bottom.text}"
    corrected_text = f"{top.text}{bottom.text}"
    validation = validate_plate_candidate(corrected_text)
    total_length = len(top.text) + len(bottom.text)
    confidence = (
        ((top.confidence * len(top.text)) + (bottom.confidence * len(bottom.text)))
        / total_length
        if total_length
        else 0.0
    )
    return Candidate(
        raw_text=raw_text,
        normalized_text=validation.normalized_text,
        ocr_confidence=confidence,
        preprocessing_variant="row_fusion",
        is_valid=validation.is_valid,
        structure_score=validation.structure_score,
        strategy="row_fusion",
        padding_ratio=0.0,
        evidence_score=(top.score + bottom.score) / 2,
        correction_used=top.correction_used or bottom.correction_used,
        correction_count=top.correction_count + bottom.correction_count,
    )


def fuse_row_candidates(
    row_candidates: list[RowCandidate],
) -> tuple[RowSelection | None, RowSelection | None, Candidate | None]:
    top = select_best_row_candidate(row_candidates, "top")
    bottom = select_best_row_candidate(row_candidates, "bottom")
    return top, bottom, _fused_candidate(top, bottom)


def _choose_two_line_winner(complete: Candidate, fused: Candidate | None) -> Candidate:
    if fused is None:
        return complete
    if fused.is_valid and not complete.is_valid:
        return fused
    if complete.is_valid and not fused.is_valid:
        return complete
    complete_score = (0.55 * complete.ocr_confidence) + (0.45 * complete.structure_score)
    fused_score = (
        0.35 * fused.ocr_confidence
        + 0.35 * fused.structure_score
        + 0.30 * fused.evidence_score
    )
    if fused.is_valid and fused.evidence_score >= 0.70 and fused_score + 0.08 >= complete_score:
        return fused
    return fused if fused_score > complete_score else complete


def _capture_variants(
    debug_images: dict[str, np.ndarray],
    padding_label: str,
    padded_crop: np.ndarray,
    variants: dict[str, np.ndarray],
) -> None:
    debug_images[f"{padding_label}_crop"] = padded_crop
    for variant_name, variant_image in variants.items():
        debug_images[f"{padding_label}_{variant_name}"] = variant_image


def _prepared_variant(
    cache: PreprocessingCache,
    variant_name: str,
    metrics: _TimingMetrics,
) -> np.ndarray:
    started = perf_counter()
    variant = cache.get(variant_name)
    metrics.preprocessing_seconds += perf_counter() - started
    return variant


def _run_full_candidate(
    cache: PreprocessingCache,
    variant_name: str,
    padding_label: str,
    padding_ratio: float,
    decoder: str,
    candidates: list[Candidate],
    rows: list[RowCandidate],
    tracker: _ProgressTracker,
    metrics: _TimingMetrics,
) -> None:
    image = _prepared_variant(cache, variant_name, metrics)
    suffix = "" if decoder == "greedy" else f"_{decoder}"
    identity = f"{padding_label}_{variant_name}{suffix}"
    result, elapsed = _timed_ocr(
        image, "full", decoder, identity, tracker, metrics
    )
    candidates.append(_candidate_from_result(result, identity, "full", padding_ratio, elapsed))
    rows.extend(
        _row_candidates_from_full_result(
            result, image.shape[0], identity, variant_name, padding_label
        )
    )


def _run_row_candidate(
    row: str,
    cache: PreprocessingCache,
    variant_name: str,
    source: str,
    decoder: str,
    rows: list[RowCandidate],
    tracker: _ProgressTracker,
    metrics: _TimingMetrics,
    padding_label: str = "padding10",
) -> None:
    image = _prepared_variant(cache, variant_name, metrics)
    label = source if decoder == "greedy" else f"{source}_{decoder}"
    result, _ = _timed_ocr(image, row, decoder, label, tracker, metrics)
    candidate = _row_candidate(
        row,
        result.raw_text,
        result.confidence,
        label,
        variant_name,
        padding_label,
        decoder,
    )
    if candidate:
        rows.append(candidate)


def _row_evidence_is_strong(
    top: RowSelection | None,
    bottom: RowSelection | None,
    fused: Candidate | None,
) -> bool:
    return bool(
        top
        and bottom
        and fused
        and fused.is_valid
        and top.score >= STRONG_TOP_SCORE
        and bottom.score >= STRONG_BOTTOM_SCORE
        and top.confidence >= 0.45
        and bottom.confidence >= 0.45
        and top_row_structure_score(top.text) >= 0.80
        and bottom_row_structure_score(bottom.text) >= 0.70
    )


def _row_weakness(selection: RowSelection | None, row: str) -> float:
    if selection is None:
        return 2.0
    if row == "top":
        structure = top_row_structure_score(selection.text)
        return (
            max(0.0, 0.80 - structure) * 2
            + max(0.0, STRONG_TOP_SCORE - selection.score)
            + max(0.0, 0.45 - selection.confidence)
        )
    structure = bottom_row_structure_score(selection.text)
    return (
        max(0.0, 0.70 - structure) * 2
        + max(0.0, STRONG_BOTTOM_SCORE - selection.score)
        + max(0.0, 0.45 - selection.confidence)
    )


def _analyze_two_line_normal(
    image: np.ndarray,
    detection: Detection,
    tracker: _ProgressTracker,
) -> tuple[list[Candidate], list[RowCandidate], _TimingMetrics]:
    metrics = _TimingMetrics()
    padding_ratio, padding_label = PADDING_OPTIONS[1]
    preprocess_started = perf_counter()
    padded_crop = crop_with_padding(image, detection.bbox, padding_ratio)
    top_crop, bottom_crop = split_two_line_crop(padded_crop)
    fallback_crop = crop_with_padding(image, detection.bbox, 0.05)
    tight_top_crop = tight_top_row_crop(fallback_crop)
    full_cache = PreprocessingCache(padded_crop)
    top_cache = PreprocessingCache(top_crop)
    tight_top_cache = PreprocessingCache(tight_top_crop)
    bottom_cache = PreprocessingCache(bottom_crop)
    metrics.preprocessing_seconds += perf_counter() - preprocess_started
    candidates: list[Candidate] = []
    rows: list[RowCandidate] = []

    # Four-call fast path: two full-crop views provide fragments while each row
    # receives its strongest inexpensive representation.
    _run_full_candidate(full_cache, "grayscale", padding_label, padding_ratio, "greedy", candidates, rows, tracker, metrics)
    _run_full_candidate(full_cache, "otsu", padding_label, padding_ratio, "greedy", candidates, rows, tracker, metrics)
    _run_row_candidate("top", top_cache, "grayscale", f"{padding_label}_split_top_grayscale", "greedy", rows, tracker, metrics)
    _run_row_candidate("bottom", bottom_cache, "otsu", f"{padding_label}_split_bottom_otsu", "greedy", rows, tracker, metrics)

    top, bottom, fusion = fuse_row_candidates(rows)
    if _row_evidence_is_strong(top, bottom, fusion):
        return candidates, rows, metrics

    weak_top = (
        top is None
        or top.score < STRONG_TOP_SCORE
        or top.confidence < 0.45
        or top_row_structure_score(top.text) < 0.80
    )
    weak_bottom = (
        bottom is None
        or bottom.score < STRONG_BOTTOM_SCORE
        or bottom.confidence < 0.45
        or bottom_row_structure_score(bottom.text) < 0.70
    )
    if weak_top and metrics.ocr_calls < MAX_NORMAL_OCR_CALLS:
        _run_row_candidate("top", tight_top_cache, "grayscale", "padding05_tight_top_grayscale", "greedy", rows, tracker, metrics, "padding05")
    elif weak_bottom and metrics.ocr_calls < MAX_NORMAL_OCR_CALLS:
        _run_row_candidate("bottom", bottom_cache, "grayscale", f"{padding_label}_split_bottom_grayscale", "greedy", rows, tracker, metrics)

    top, bottom, fusion = fuse_row_candidates(rows)
    if _row_evidence_is_strong(top, bottom, fusion):
        return candidates, rows, metrics

    weak_top = (
        top is None
        or top.score < STRONG_TOP_SCORE
        or top.confidence < 0.45
        or top_row_structure_score(top.text) < 0.80
    )
    weak_bottom = (
        bottom is None
        or bottom.score < STRONG_BOTTOM_SCORE
        or bottom.confidence < 0.45
        or bottom_row_structure_score(bottom.text) < 0.70
    )
    if metrics.ocr_calls < MAX_NORMAL_OCR_CALLS:
        if weak_top and (not weak_bottom or _row_weakness(top, "top") >= _row_weakness(bottom, "bottom")):
            _run_row_candidate("top", top_cache, "sharpened_grayscale", f"{padding_label}_split_top_sharpened_grayscale", "beamsearch", rows, tracker, metrics)
        elif weak_bottom:
            _run_row_candidate("bottom", bottom_cache, "grayscale", f"{padding_label}_split_bottom_grayscale", "greedy", rows, tracker, metrics)
    return candidates, rows, metrics


def _analyze_standard_normal(
    image: np.ndarray,
    detection: Detection,
    tracker: _ProgressTracker,
) -> tuple[list[Candidate], _TimingMetrics]:
    metrics = _TimingMetrics()
    candidates: list[Candidate] = []
    sharpened_image = None
    for padding_label, variant_names in ONE_LINE_NORMAL_PLAN.items():
        padding_ratio = 0.10
        started = perf_counter()
        cache = PreprocessingCache(crop_with_padding(image, detection.bbox, padding_ratio))
        metrics.preprocessing_seconds += perf_counter() - started
        for variant_name in variant_names:
            identity = f"{padding_label}_{variant_name}"
            variant_image = _prepared_variant(cache, variant_name, metrics)
            result, elapsed = _timed_ocr(
                variant_image, "full", "greedy", identity, tracker, metrics
            )
            candidates.append(
                _candidate_from_result(result, identity, "full", padding_ratio, elapsed)
            )
            if variant_name == "sharpened_grayscale":
                sharpened_image = variant_image

    winner = select_best_candidate(candidates)
    if sharpened_image is not None and (not winner.is_valid or winner.ocr_confidence < 0.45):
        result, elapsed = _timed_ocr(
            sharpened_image,
            "full",
            "beamsearch",
            "padding10_sharpened_grayscale_beamsearch",
            tracker,
            metrics,
        )
        candidates.append(
            _candidate_from_result(
                result,
                "padding10_sharpened_grayscale_beamsearch",
                "full",
                0.10,
                elapsed,
            )
        )
    return candidates, metrics


def _analyze_exhaustive(
    image: np.ndarray,
    detection: Detection,
    likely_two_line: bool,
    tracker: _ProgressTracker,
    capture_images: bool,
) -> tuple[list[Candidate], list[RowCandidate], _TimingMetrics, dict[str, np.ndarray]]:
    metrics = _TimingMetrics()
    candidates: list[Candidate] = []
    rows: list[RowCandidate] = []
    debug_images: dict[str, np.ndarray] = {}
    if capture_images:
        debug_images["crop"] = crop_to_bbox(image, detection.bbox)

    for padding_ratio, padding_label in PADDING_OPTIONS:
        preprocessing_started = perf_counter()
        padded_crop = crop_with_padding(image, detection.bbox, padding_ratio)
        variants = preprocessing_variants(padded_crop)
        if capture_images:
            _capture_variants(debug_images, padding_label, padded_crop, variants)

        top_crop = bottom_crop = None
        top_variants = bottom_variants = None
        if likely_two_line:
            top_crop, bottom_crop = split_two_line_crop(padded_crop)
            top_variants = preprocessing_variants(top_crop)
            bottom_variants = preprocessing_variants(bottom_crop)
            if capture_images:
                debug_images[f"{padding_label}_split_top_crop"] = top_crop
                debug_images[f"{padding_label}_split_bottom_crop"] = bottom_crop
        metrics.preprocessing_seconds += perf_counter() - preprocessing_started

        for variant_name in ALL_VARIANTS:
            identity = f"{padding_label}_{variant_name}"
            result, elapsed = _timed_ocr(
                variants[variant_name], "full", "beamsearch", identity, tracker, metrics
            )
            candidates.append(
                _candidate_from_result(result, identity, "full", padding_ratio, elapsed)
            )
            if likely_two_line:
                rows.extend(
                    _row_candidates_from_full_result(
                        result,
                        variants[variant_name].shape[0],
                        identity,
                        variant_name,
                        padding_label,
                    )
                )
                top_identity = f"{padding_label}_split_top_{variant_name}"
                bottom_identity = f"{padding_label}_split_bottom_{variant_name}"
                top_result, top_elapsed = _timed_ocr(
                    top_variants[variant_name], "top", "beamsearch", top_identity, tracker, metrics
                )
                bottom_result, bottom_elapsed = _timed_ocr(
                    bottom_variants[variant_name],
                    "bottom",
                    "beamsearch",
                    bottom_identity,
                    tracker,
                    metrics,
                )
                combined = combine_ocr_results(top_result, bottom_result)
                candidates.append(
                    _candidate_from_result(
                        combined,
                        f"{padding_label}_split_rows_{variant_name}",
                        "split_rows",
                        padding_ratio,
                        top_elapsed + bottom_elapsed,
                    )
                )
                for row, row_result, source in (
                    ("top", top_result, top_identity),
                    ("bottom", bottom_result, bottom_identity),
                ):
                    row_candidate = _row_candidate(
                        row,
                        row_result.raw_text,
                        row_result.confidence,
                        source,
                        variant_name,
                        padding_label,
                        "beamsearch",
                    )
                    if row_candidate:
                        rows.append(row_candidate)
                if capture_images:
                    debug_images[top_identity] = top_variants[variant_name]
                    debug_images[bottom_identity] = bottom_variants[variant_name]
    return candidates, rows, metrics, debug_images


def _analyze_detection(
    image: np.ndarray,
    detection: Detection,
    exhaustive: bool,
    capture_images: bool,
    tracker: _ProgressTracker,
) -> DetectionAnalysis:
    analysis_started = perf_counter()
    quality = assess_detection_quality(detection)
    if not quality.should_run:
        return DetectionAnalysis(
            detection=detection,
            winner=_empty_candidate(),
            candidates=[],
            split_row_used=False,
            row_fusion_used=False,
            top_row=None,
            bottom_row=None,
            row_candidates=[],
            ocr_execution_seconds=0.0,
            ocr_status=quality.status,
            total_recognition_seconds=perf_counter() - analysis_started,
            debug_images={"crop": crop_to_bbox(image, detection.bbox)} if capture_images else {},
        )

    raw_crop = crop_to_bbox(image, detection.bbox)
    height, width = raw_crop.shape[:2]
    likely_two_line = width > 0 and (height / width) >= 0.65

    if exhaustive:
        candidates, rows, metrics, debug_images = _analyze_exhaustive(
            image, detection, likely_two_line, tracker, capture_images
        )
    elif likely_two_line:
        candidates, rows, metrics = _analyze_two_line_normal(image, detection, tracker)
        debug_images = {}
    else:
        candidates, metrics = _analyze_standard_normal(image, detection, tracker)
        rows = []
        debug_images = {}

    ranking_started = perf_counter()
    best_complete = select_best_candidate(candidates)
    if likely_two_line:
        top, bottom, fusion = fuse_row_candidates(rows)
    else:
        top = bottom = fusion = None
    if fusion:
        candidates.append(fusion)
    winner = _choose_two_line_winner(best_complete, fusion) if likely_two_line else best_complete
    metrics.ranking_seconds += perf_counter() - ranking_started
    return DetectionAnalysis(
        detection=detection,
        winner=winner,
        candidates=candidates,
        split_row_used=likely_two_line,
        row_fusion_used=winner.strategy == "row_fusion",
        top_row=top,
        bottom_row=bottom,
        row_candidates=rows,
        ocr_execution_seconds=metrics.ocr_execution_seconds,
        ocr_status="ok" if winner.is_valid else "unreadable",
        correction_used=winner.correction_used,
        correction_count=winner.correction_count,
        ocr_calls=metrics.ocr_calls,
        decoders_used=tuple(dict.fromkeys(metrics.decoders)),
        preprocessing_seconds=metrics.preprocessing_seconds,
        ranking_seconds=metrics.ranking_seconds,
        total_recognition_seconds=perf_counter() - analysis_started,
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
    progress_callback: ProgressCallback | None = None,
) -> list[DetectionAnalysis]:
    recognition_started = perf_counter()
    yolo_started = perf_counter()
    detections = detect_plates(image, confidence_threshold)
    yolo_elapsed = perf_counter() - yolo_started
    if exhaustive:
        total = 0
        for detection in detections:
            if not assess_detection_quality(detection).should_run:
                continue
            crop = crop_to_bbox(image, detection.bbox)
            height, width = crop.shape[:2]
            total += 63 if width > 0 and (height / width) >= 0.65 else 21
    else:
        total = 0
    tracker = _ProgressTracker(total=total, callback=progress_callback)
    analyses = [
        _analyze_detection(image, detection, exhaustive, capture_images, tracker)
        for detection in detections
    ]
    analyses.sort(key=_analysis_ranking_key, reverse=True)
    total_elapsed = perf_counter() - recognition_started
    for analysis in analyses:
        analysis.yolo_inference_seconds = yolo_elapsed
        analysis.total_recognition_seconds = total_elapsed
    return analyses


def recognize_license_plates(image: np.ndarray, confidence_threshold: float) -> list[dict[str, object]]:
    return [
        analysis.public_result()
        for analysis in analyze_license_plates(image, confidence_threshold)
    ]
