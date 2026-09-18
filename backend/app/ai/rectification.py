import cv2
import numpy as np


def _ordered_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def rectify_plate_crop(crop: np.ndarray) -> tuple[np.ndarray, bool]:
    """Apply one conservative four-corner warp, or return the original safely."""
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 40:
        return crop, False
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = crop.shape[0] * crop.shape[1]
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        quadrilateral = None
        for contour in candidates:
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            coverage = cv2.contourArea(approximation) / image_area
            if len(approximation) == 4 and 0.20 <= coverage <= 0.98:
                quadrilateral = approximation.reshape(4, 2).astype(np.float32)
                break
        if quadrilateral is None:
            return crop, False

        top_left, top_right, bottom_right, bottom_left = _ordered_points(quadrilateral)
        top = np.linalg.norm(top_right - top_left)
        bottom = np.linalg.norm(bottom_right - bottom_left)
        left = np.linalg.norm(bottom_left - top_left)
        right = np.linalg.norm(bottom_right - top_right)
        if min(top, bottom, left, right) < 5:
            return crop, False
        perspective_ratio = max(top / bottom, bottom / top, left / right, right / left)
        top_angle = abs(np.degrees(np.arctan2(*(top_right - top_left)[::-1])))
        top_angle = min(top_angle, abs(180 - top_angle))
        if perspective_ratio < 1.08 and top_angle < 3.0:
            return crop, False

        width = int(round(max(top, bottom)))
        height = int(round(max(left, right)))
        if width < 40 or height < 20 or width / height < 1.1:
            return crop, False
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(
            np.array([top_left, top_right, bottom_right, bottom_left]),
            destination,
        )
        rectified = cv2.warpPerspective(crop, matrix, (width, height))
        return (rectified, True) if rectified.size else (crop, False)
    except (cv2.error, ValueError, ZeroDivisionError):
        return crop, False
