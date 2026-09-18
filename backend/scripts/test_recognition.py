"""Run the production image-recognition pipeline against one local image."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
from sqlalchemy import select


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ai.pipeline import analyze_license_plates  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.database.init_db import initialize_database  # noqa: E402
from app.models import Setting  # noqa: E402


DEBUG_OUTPUT_DIR = BACKEND_DIR / "debug_output"


def _save_debug_images(detection_index: int, images: dict) -> None:
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_name, image in images.items():
        output_path = DEBUG_OUTPUT_DIR / f"detection_{detection_index}_{image_name}.jpg"
        if not cv2.imwrite(str(output_path), image):
            print(f"Warning: could not save {output_path}", file=sys.stderr)


def _print_debug_analysis(index: int, analysis) -> None:
    print(f"\nDetection {index}")
    print(f"Detection confidence: {analysis.detection.confidence:.4f}")
    print(f"Bounding box: {analysis.detection.bbox}")
    print(f"OCR status: {analysis.ocr_status}")
    print(f"Split-row OCR used: {analysis.split_row_used}")
    print("Row candidates:")
    for row_candidate in analysis.row_candidates:
        print(
            "  "
            f"{row_candidate.row.upper()}: {row_candidate.normalized_text} "
            f"(confidence={row_candidate.confidence:.4f}, source={row_candidate.source})"
        )
    print("OCR candidates:")
    for candidate in analysis.candidates:
        fragments = [asdict(fragment) for fragment in candidate.fragments]
        print(f"\n{candidate.preprocessing_variant}:")
        print(f"  strategy: {candidate.strategy}")
        print(f"  fragments: {json.dumps(fragments, ensure_ascii=False)}")
        print(f"  raw: {candidate.raw_text}")
        print(f"  normalized: {candidate.normalized_text}")
        print(f"  confidence: {candidate.ocr_confidence:.4f}")
        print(f"  structure_score: {candidate.structure_score:.2f}")
        print(f"  valid: {str(candidate.is_valid).lower()}")
        print(f"  OCR time: {candidate.execution_seconds:.3f}s")

    print("\nFinal winner:")
    print(json.dumps(analysis.public_result(), indent=2))
    print(f"Winning strategy: {analysis.winner.strategy}")
    print(f"Winning preprocessing: {analysis.winner.preprocessing_variant}")
    print(f"Best top row: {analysis.top_row_text} ({analysis.top_row_confidence:.4f})")
    print(f"Best bottom row: {analysis.bottom_row_text} ({analysis.bottom_row_confidence:.4f})")
    print(f"Row fusion used: {analysis.row_fusion_used}")
    print(f"Correction used: {analysis.correction_used}")
    print(f"Correction count: {analysis.correction_count}")
    print(f"OCR engine: {analysis.ocr_engine}")
    print(f"EasyOCR fallback used: {analysis.fallback_used}")
    if analysis.fallback_reasons:
        print(f"Fallback reasons: {', '.join(analysis.fallback_reasons)}")
    print(f"OCR calls: {analysis.ocr_calls}")
    print(f"Decoders used: {', '.join(analysis.decoders_used) or 'none'}")
    print(f"YOLO inference time: {analysis.yolo_inference_seconds:.3f}s")
    print(f"Preprocessing time: {analysis.preprocessing_seconds:.3f}s")
    print(f"Detection OCR execution time: {analysis.ocr_execution_seconds:.3f}s")
    print(f"Paddle primary time: {analysis.primary_ocr_seconds:.3f}s")
    print(f"EasyOCR fallback time: {analysis.fallback_ocr_seconds:.3f}s")
    print(f"Ranking/fusion time: {analysis.ranking_seconds:.3f}s")
    print(f"Total recognition time: {analysis.total_recognition_seconds:.3f}s")
    _save_debug_images(index, analysis.debug_images)


def _print_progress(completed: int, total: int, label: str, elapsed: float) -> None:
    print(f"[{completed}/{total}] {label} ({elapsed:.1f}s elapsed)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize license plates in an image.")
    parser.add_argument("image", type=Path, help="Path to a JPEG or PNG image")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run exhaustive OCR candidates and save intermediate images",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Unable to decode image: {image_path}", file=sys.stderr)
        return 1

    initialize_database()
    with SessionLocal() as db:
        settings = db.scalar(select(Setting).limit(1))
        confidence = float(settings.yolo_confidence)

    started = perf_counter()
    if args.debug:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        analyses = analyze_license_plates(
            image,
            confidence,
            exhaustive=True,
            capture_images=True,
            progress_callback=_print_progress,
        )
        print(f"YOLO completed: {len(analyses)} detections", flush=True)
        for index, analysis in enumerate(analyses, start=1):
            _print_debug_analysis(index, analysis)
        print(f"\nTotal recognition time: {perf_counter() - started:.3f}s")
        print(f"Debug images: {DEBUG_OUTPUT_DIR}")
    else:
        analyses = analyze_license_plates(image, confidence)
        print(f"Detections: {len(analyses)}")
        for index, analysis in enumerate(analyses, start=1):
            print(f"\nDetection {index}")
            print(json.dumps(analysis.public_result(), indent=2))
            print(f"Best top row: {analysis.top_row_text} ({analysis.top_row_confidence:.4f})")
            print(f"Best bottom row: {analysis.bottom_row_text} ({analysis.bottom_row_confidence:.4f})")
            print(f"Row fusion used: {analysis.row_fusion_used}")
            print(f"OCR status: {analysis.ocr_status}")
            print(f"Correction used: {analysis.correction_used} ({analysis.correction_count})")
            print(f"OCR engine: {analysis.ocr_engine}")
            print(f"EasyOCR fallback used: {analysis.fallback_used}")
            if analysis.fallback_reasons:
                print(f"Fallback reasons: {', '.join(analysis.fallback_reasons)}")
            print(f"OCR calls: {analysis.ocr_calls}")
            print(f"Decoders used: {', '.join(analysis.decoders_used) or 'none'}")
            print(f"YOLO inference time: {analysis.yolo_inference_seconds:.3f}s")
            print(f"Preprocessing time: {analysis.preprocessing_seconds:.3f}s")
            print(f"OCR execution time: {analysis.ocr_execution_seconds:.3f}s")
            print(f"Paddle primary time: {analysis.primary_ocr_seconds:.3f}s")
            print(f"EasyOCR fallback time: {analysis.fallback_ocr_seconds:.3f}s")
            print(f"Ranking/fusion time: {analysis.ranking_seconds:.3f}s")
        print(f"\nTotal recognition time: {perf_counter() - started:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
