"""Reproducible detector/OCR benchmark for labeled repository images."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ai.evaluation import (  # noqa: E402
    EvaluationRow,
    levenshtein_distance,
    make_evaluation_row,
    summarize_evaluations,
)
from app.ai.frame_quality import calculate_sharpness  # noqa: E402
from app.ai.ocr_engines import OCREngineUnavailable, recognize_with_paddleocr  # noqa: E402
from app.ai.pipeline import DetectionAnalysis, analyze_license_plates  # noqa: E402
from app.ai.preprocess import crop_with_padding  # noqa: E402
from app.ai.rectification import rectify_plate_crop  # noqa: E402
from app.ai.validator import normalize_ocr_text  # noqa: E402


DEFAULT_DATASET = REPOSITORY_DIR / "images"
DEFAULT_LABELS = BACKEND_DIR / "benchmark" / "ground_truth.csv"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--compare-paddle", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _load_labels(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").strip()
            expected = normalize_ocr_text(
                row.get("plate") or row.get("expected_plate") or ""
            )
            if not filename or not expected:
                raise ValueError("Every CSV row requires filename and plate")
            grouped[filename].append(expected)
    if not grouped:
        raise ValueError("Ground-truth CSV has no labeled rows")
    return dict(grouped)


def _assign_analyses(
    expected_plates: list[str], analyses: list[DetectionAnalysis]
) -> list[DetectionAnalysis | None]:
    available = list(analyses)
    assigned: list[DetectionAnalysis | None] = []
    for expected in expected_plates:
        if not available:
            assigned.append(None)
            continue
        best_index = min(
            range(len(available)),
            key=lambda index: levenshtein_distance(
                expected, available[index].winner.normalized_text
            ),
        )
        assigned.append(available.pop(best_index))
    return assigned


def _quality_issue(image, analysis: DetectionAnalysis | None) -> str:
    if analysis is None:
        return ""
    crop = crop_with_padding(image, analysis.detection.bbox, 0.05)
    if calculate_sharpness(crop) < 30:
        return "blur"
    _, rectification_applicable = rectify_plate_crop(crop)
    return "perspective" if rectification_applicable else ""


def _easyocr_rows(
    filename: str,
    expected_plates: list[str],
    image,
    analyses: list[DetectionAnalysis],
    measured_seconds: float,
) -> list[EvaluationRow]:
    rows = []
    yolo_seconds = analyses[0].yolo_inference_seconds if analyses else measured_seconds
    for expected, analysis in zip(expected_plates, _assign_analyses(expected_plates, analyses)):
        winner = analysis.winner if analysis else None
        rows.append(
            make_evaluation_row(
                filename=filename,
                expected=expected,
                predicted=winner.normalized_text if winner else "",
                detected=analysis is not None,
                detection_confidence=analysis.detection.confidence if analysis else 0.0,
                ocr_confidence=winner.ocr_confidence if winner else 0.0,
                ocr_status=analysis.ocr_status if analysis else "detector_miss",
                yolo_seconds=analysis.yolo_inference_seconds if analysis else yolo_seconds,
                quality_seconds=analysis.quality_gate_seconds if analysis else 0.0,
                ocr_seconds=analysis.ocr_execution_seconds if analysis else 0.0,
                total_seconds=analysis.total_recognition_seconds if analysis else measured_seconds,
                quality_issue=_quality_issue(image, analysis),
                bbox=analysis.detection.bbox if analysis else None,
                raw_ocr=winner.raw_text if winner else "",
                ocr_valid=winner.is_valid if winner else False,
            )
        )
    return rows


def _paddle_rows(
    filename: str,
    expected_plates: list[str],
    image,
    analyses: list[DetectionAnalysis],
) -> list[EvaluationRow]:
    engine_results = []
    for analysis in analyses:
        crop = crop_with_padding(image, analysis.detection.bbox, 0.10)
        started = perf_counter()
        result = recognize_with_paddleocr(crop)
        engine_results.append((analysis, result, perf_counter() - started))
    available = list(engine_results)
    rows = []
    for expected in expected_plates:
        if not available:
            rows.append(
                make_evaluation_row(
                    filename=filename,
                    expected=expected,
                    predicted="",
                    detected=False,
                    ocr_status="detector_miss",
                    engine="paddleocr",
                )
            )
            continue
        index = min(
            range(len(available)),
            key=lambda item: levenshtein_distance(
                expected, available[item][1].normalized_text
            ),
        )
        analysis, result, elapsed = available.pop(index)
        rows.append(
            make_evaluation_row(
                filename=filename,
                expected=expected,
                predicted=result.normalized_text,
                detected=True,
                detection_confidence=analysis.detection.confidence,
                ocr_confidence=result.confidence,
                ocr_status="ok" if result.is_valid else "unreadable",
                yolo_seconds=analysis.yolo_inference_seconds,
                quality_seconds=analysis.quality_gate_seconds,
                ocr_seconds=elapsed,
                total_seconds=analysis.yolo_inference_seconds + elapsed,
                engine="paddleocr",
                quality_issue=_quality_issue(image, analysis),
                bbox=analysis.detection.bbox,
                raw_ocr=result.raw_text,
                ocr_valid=result.is_valid,
            )
        )
    return rows


def _print_rows(rows: list[EvaluationRow]) -> None:
    print(
        "engine    file           expected     raw/predicted       det exact char  "
        "yolo_conf ocr_conf yolo_ms ocr_ms total_ms error"
    )
    for row in rows:
        print(
            f"{row.engine:<9} {row.filename:<14} {row.expected:<12} "
            f"{((row.raw_ocr or '-') + '/' + (row.predicted or '-')): <19} {str(row.detected):<3} "
            f"{str(row.exact_match):<5} {row.character_accuracy:>4.2f} "
            f"{row.detection_confidence:>9.3f} {row.ocr_confidence:>8.3f} "
            f"{row.yolo_seconds * 1000:>7.1f} {row.ocr_seconds * 1000:>6.1f} "
            f"{row.total_seconds * 1000:>8.1f} {row.error_category}"
        )


def main() -> int:
    args = _arguments()
    labels = _load_labels(args.ground_truth.resolve())
    dataset = args.dataset.resolve()
    images = {}
    for filename in labels:
        path = dataset / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read labeled image: {path}")
        images[filename] = image

    if not args.no_warmup:
        first_image = images[next(iter(images))]
        analyze_license_plates(first_image, args.confidence)

    rows: list[EvaluationRow] = []
    paddle_rows: list[EvaluationRow] = []
    for filename, expected_plates in labels.items():
        image = images[filename]
        started = perf_counter()
        analyses = analyze_license_plates(image, args.confidence)
        measured_seconds = perf_counter() - started
        rows.extend(
            _easyocr_rows(
                filename, expected_plates, image, analyses, measured_seconds
            )
        )
        if args.compare_paddle:
            try:
                paddle_rows.extend(_paddle_rows(filename, expected_plates, image, analyses))
            except OCREngineUnavailable as exc:
                print(str(exc), file=sys.stderr)
                return 2

    all_rows = rows + paddle_rows
    summaries = {"easyocr": summarize_evaluations(rows)}
    if paddle_rows:
        summaries["paddleocr"] = summarize_evaluations(paddle_rows)
    _print_rows(all_rows)
    print(json.dumps(summaries, indent=2))

    if args.output_json:
        payload = {
            "configuration": {
                "dataset": str(dataset),
                "ground_truth": str(args.ground_truth.resolve()),
                "confidence": args.confidence,
                "warmup": not args.no_warmup,
            },
            "rows": [row.as_dict() for row in all_rows],
            "summary": summaries,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
