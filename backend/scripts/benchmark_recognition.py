"""Reproducible detector/OCR benchmark for labeled repository images."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable

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
from app.ai.detector import detect_plates  # noqa: E402
from app.ai.frame_quality import calculate_sharpness  # noqa: E402
from app.ai.ocr_engines import (  # noqa: E402
    OCREngineUnavailable,
    initialize_paddleocr,
    paddle_runtime_diagnostics,
)
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
    parser.add_argument(
        "--engine",
        choices=("easyocr", "paddleocr", "production"),
        default="production",
        help="Run exactly one OCR mode per process (default: production)",
    )
    parser.add_argument(
        "--compare-paddle",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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


def _evaluation_rows(
    filename: str,
    expected_plates: list[str],
    image,
    analyses: list[DetectionAnalysis],
    measured_seconds: float,
    benchmark_engine: str,
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
                engine=benchmark_engine,
                primary_ocr_seconds=(analysis.primary_ocr_seconds if analysis else 0.0),
                fallback_ocr_seconds=(analysis.fallback_ocr_seconds if analysis else 0.0),
                fallback_used=analysis.fallback_used if analysis else False,
            )
        )
    return rows


def _print_rows(rows: list[EvaluationRow]) -> None:
    print(
        "engine     file           expected     raw/predicted       det exact fb char  "
        "yolo_conf ocr_conf yolo_ms primary_ms fallback_ms total_ms error"
    )
    for row in rows:
        print(
            f"{row.engine:<10} {row.filename:<14} {row.expected:<12} "
            f"{((row.raw_ocr or '-') + '/' + (row.predicted or '-')): <19} {str(row.detected):<3} "
            f"{str(row.exact_match):<5} {str(row.fallback_used):<2} {row.character_accuracy:>4.2f} "
            f"{row.detection_confidence:>9.3f} {row.ocr_confidence:>8.3f} "
            f"{row.yolo_seconds * 1000:>7.1f} {row.primary_ocr_seconds * 1000:>10.1f} "
            f"{row.fallback_ocr_seconds * 1000:>11.1f} "
            f"{row.total_seconds * 1000:>8.1f} {row.error_category}"
        )


def _initialize_torch_runtime() -> None:
    # Deliberately local: benchmark import remains lightweight, while Windows
    # loads PyTorch native DLLs before Ultralytics and PaddlePaddle.
    import torch  # noqa: F401


def _warm_yolo_runtime(image, confidence: float) -> None:
    detect_plates(image, confidence)


def _initialize_benchmark_runtimes(
    first_image,
    confidence: float,
    *,
    use_paddle: bool,
    warmup: bool,
    torch_initialize: Callable = _initialize_torch_runtime,
    analyze: Callable = _warm_yolo_runtime,
    paddle_diagnostics: Callable = paddle_runtime_diagnostics,
    paddle_initialize: Callable = initialize_paddleocr,
    emit: Callable[[str], None] = print,
) -> None:
    # On Windows, torch/Ultralytics must own their native DLL load order before
    # anything imports PaddlePaddle. Paddle modes therefore force warmup
    # even when --no-warmup was supplied.
    if warmup or use_paddle:
        emit("Initializing PyTorch/YOLO...")
        torch_initialize()
        analyze(first_image, confidence)
        emit("PyTorch/YOLO initialized")

    if use_paddle:
        for label, value in paddle_diagnostics().items():
            emit(f"{label}: {value}")
        paddle_initialize()
        emit("PaddleOCR initialized")


def main() -> int:
    args = _arguments()
    benchmark_engine = "paddleocr" if args.compare_paddle else args.engine
    if args.compare_paddle:
        print(
            "--compare-paddle is deprecated; running isolated --engine paddleocr",
            file=sys.stderr,
        )
    labels = _load_labels(args.ground_truth.resolve())
    dataset = args.dataset.resolve()
    images = {}
    for filename in labels:
        path = dataset / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read labeled image: {path}")
        images[filename] = image

    first_image = images[next(iter(images))]
    try:
        _initialize_benchmark_runtimes(
            first_image,
            args.confidence,
            use_paddle=benchmark_engine in {"paddleocr", "production"},
            warmup=not args.no_warmup,
            emit=lambda message: print(message, file=sys.stderr),
        )
    except OCREngineUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rows: list[EvaluationRow] = []
    for filename, expected_plates in labels.items():
        image = images[filename]
        started = perf_counter()
        analyses = analyze_license_plates(
            image, args.confidence, engine=benchmark_engine
        )
        measured_seconds = perf_counter() - started
        rows.extend(
            _evaluation_rows(
                filename,
                expected_plates,
                image,
                analyses,
                measured_seconds,
                benchmark_engine,
            )
        )

    summaries = {benchmark_engine: summarize_evaluations(rows)}
    _print_rows(rows)
    print(json.dumps(summaries, indent=2))

    if args.output_json:
        payload = {
            "configuration": {
                "dataset": str(dataset),
                "ground_truth": str(args.ground_truth.resolve()),
                "confidence": args.confidence,
                "warmup": (
                    not args.no_warmup
                    or benchmark_engine in {"paddleocr", "production"}
                ),
                "engine": benchmark_engine,
            },
            "rows": [row.as_dict() for row in rows],
            "summary": summaries,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
