"""Run the production image-recognition pipeline against one local image."""

import argparse
import json
import sys
from pathlib import Path

import cv2
from sqlalchemy import select


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ai.pipeline import recognize_license_plates  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.database.init_db import initialize_database  # noqa: E402
from app.models import Setting  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize license plates in an image.")
    parser.add_argument("image", type=Path, help="Path to a JPEG or PNG image")
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

    detections = recognize_license_plates(image, confidence)
    print(f"Detections: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        print(f"\nDetection {index}")
        print(json.dumps(detection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
