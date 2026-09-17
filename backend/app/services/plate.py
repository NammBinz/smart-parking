import re


def normalize_plate(plate_number: str) -> str:
    normalized = re.sub(r"[\s.\-]+", "", plate_number.strip().upper())
    if not normalized:
        raise ValueError("Plate number is required")
    return normalized
