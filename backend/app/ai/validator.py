import re
from dataclasses import dataclass

from app.services.plate import normalize_plate


@dataclass(frozen=True)
class ValidationResult:
    normalized_text: str
    is_valid: bool


def normalize_ocr_text(text: str) -> str:
    normalized = normalize_plate(text) if text.strip() else ""
    return re.sub(r"[^A-Z0-9]", "", normalized)


def validate_plate_candidate(text: str) -> ValidationResult:
    normalized = normalize_ocr_text(text)
    letters = sum(character.isalpha() for character in normalized)
    digits = sum(character.isdigit() for character in normalized)
    is_valid = bool(
        7 <= len(normalized) <= 10
        and re.fullmatch(r"[A-Z0-9]+", normalized)
        and letters >= 1
        and digits >= 2
    )
    return ValidationResult(normalized_text=normalized, is_valid=is_valid)
