from dataclasses import dataclass

from app.ai.detector import Detection


# Deliberately conservative and module-level so a later settings integration can
# override these without changing recognition logic.
MIN_OCR_WIDTH = 40
MIN_OCR_HEIGHT = 20


@dataclass(frozen=True)
class OCRQuality:
    width: int
    height: int
    status: str

    @property
    def should_run(self) -> bool:
        return self.status == "ready"


def assess_detection_quality(
    detection: Detection,
    min_width: int = MIN_OCR_WIDTH,
    min_height: int = MIN_OCR_HEIGHT,
) -> OCRQuality:
    x1, y1, x2, y2 = detection.bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    status = "ready" if width >= min_width and height >= min_height else "too_small"
    return OCRQuality(width=width, height=height, status=status)
