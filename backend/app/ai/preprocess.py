import cv2
import numpy as np


class PreprocessingCache:
    """Lazily build OCR representations once for a single crop."""

    def __init__(self, crop: np.ndarray):
        self.crop = crop
        self._variants: dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        if self.crop.size == 0:
            raise ValueError("Cannot preprocess an empty crop")
        if name in self._variants:
            return self._variants[name]

        if "upscaled_color" not in self._variants:
            self._variants["upscaled_color"] = cv2.resize(
                self.crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )
        if name == "upscaled_color":
            return self._variants[name]

        if "grayscale" not in self._variants:
            self._variants["grayscale"] = cv2.cvtColor(
                self._variants["upscaled_color"], cv2.COLOR_BGR2GRAY
            )
        if name == "grayscale":
            return self._variants[name]

        grayscale = self._variants["grayscale"]
        if name == "contrast_enhanced":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self._variants[name] = clahe.apply(grayscale)
        elif name == "adaptive_threshold":
            contrast = self.get("contrast_enhanced")
            self._variants[name] = cv2.adaptiveThreshold(
                contrast,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                13,
                5,
            )
        elif name == "otsu":
            _, self._variants[name] = cv2.threshold(
                grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif name == "bilateral_otsu":
            bilateral = cv2.bilateralFilter(grayscale, 7, 50, 50)
            _, self._variants[name] = cv2.threshold(
                bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif name == "sharpened_grayscale":
            self._variants[name] = cv2.filter2D(
                grayscale,
                -1,
                np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            )
        else:
            raise KeyError(f"Unknown preprocessing variant: {name}")
        return self._variants[name]

    def selected(self, names: tuple[str, ...]) -> dict[str, np.ndarray]:
        return {name: self.get(name) for name in names}


def crop_to_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))
    return image[y1:y2, x1:x2].copy()


def crop_with_padding(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding_ratio: float = 0.05,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    padding_x = max(1, int(round((x2 - x1) * padding_ratio)))
    padding_y = max(1, int(round((y2 - y1) * padding_ratio)))
    padded_x1 = max(0, x1 - padding_x)
    padded_y1 = max(0, y1 - padding_y)
    padded_x2 = min(image_width, x2 + padding_x)
    padded_y2 = min(image_height, y2 + padding_y)
    return image[padded_y1:padded_y2, padded_x1:padded_x2].copy()


def preprocessing_variants(crop: np.ndarray) -> dict[str, np.ndarray]:
    if crop.size == 0:
        return {}
    cache = PreprocessingCache(crop)
    names = (
        "upscaled_color",
        "grayscale",
        "contrast_enhanced",
        "adaptive_threshold",
        "otsu",
        "bilateral_otsu",
        "sharpened_grayscale",
    )
    return cache.selected(names)


def split_two_line_crop(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = crop.shape[0]
    top_end = max(1, min(height, int(round(height * 0.55))))
    bottom_start = max(0, min(height - 1, int(round(height * 0.45))))
    return crop[:top_end].copy(), crop[bottom_start:].copy()


def tight_top_row_crop(crop: np.ndarray) -> np.ndarray:
    """Exclude the upper border and most of the bottom row for a top-row retry."""
    height = crop.shape[0]
    start = max(0, min(height - 1, int(round(height * 0.05))))
    end = max(start + 1, min(height, int(round(height * 0.48))))
    return crop[start:end].copy()
