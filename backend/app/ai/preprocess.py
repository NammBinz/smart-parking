import cv2
import numpy as np


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
    upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(grayscale)
    adaptive = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        13,
        5,
    )
    _, otsu = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    bilateral = cv2.bilateralFilter(grayscale, 7, 50, 50)
    _, bilateral_otsu = cv2.threshold(
        bilateral,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    sharpened = cv2.filter2D(
        grayscale,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    return {
        "upscaled_color": upscaled,
        "grayscale": grayscale,
        "contrast_enhanced": contrast,
        "adaptive_threshold": adaptive,
        "otsu": otsu,
        "bilateral_otsu": bilateral_otsu,
        "sharpened_grayscale": sharpened,
    }


def split_two_line_crop(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = crop.shape[0]
    top_end = max(1, min(height, int(round(height * 0.55))))
    bottom_start = max(0, min(height - 1, int(round(height * 0.45))))
    return crop[:top_end].copy(), crop[bottom_start:].copy()
