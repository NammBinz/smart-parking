import re
from dataclasses import dataclass

from app.services.plate import normalize_plate


@dataclass(frozen=True)
class ValidationResult:
    normalized_text: str
    is_valid: bool
    structure_score: float


def normalize_ocr_text(text: str) -> str:
    normalized = normalize_plate(text) if text.strip() else ""
    return re.sub(r"[^A-Z0-9]", "", normalized)


def plate_structure_score(normalized_text: str) -> float:
    if not normalized_text or not re.fullmatch(r"[A-Z0-9]+", normalized_text):
        return 0.0

    length = len(normalized_text)
    letter_count = sum(character.isalpha() for character in normalized_text)
    digit_count = sum(character.isdigit() for character in normalized_text)
    score = 0.0
    if 7 <= length <= 10:
        score += 0.25
    if length >= 2 and normalized_text[:2].isdigit():
        score += 0.25
    if any(character.isalpha() for character in normalized_text[2:5]):
        score += 0.25
    if 1 <= letter_count <= 3:
        score += 0.15
    if digit_count >= 5 and digit_count > letter_count:
        score += 0.10
    return min(1.0, score)


def plate_layout_score(text: str) -> float:
    """Softly score common Vietnamese plate layout without repairing characters."""
    normalized = normalize_ocr_text(text)
    if not normalized:
        return 0.0
    score = 0.0
    if 7 <= len(normalized) <= 10:
        score += 0.20
    if len(normalized) >= 2 and normalized[:2].isdigit():
        score += 0.25
    if len(normalized) >= 3 and normalized[2].isalpha():
        score += 0.30
    if len(normalized) >= 4 and normalized[-4:].isdigit():
        score += 0.15
    letters = sum(character.isalpha() for character in normalized)
    digits = sum(character.isdigit() for character in normalized)
    if 1 <= letters <= 3 and digits >= 5:
        score += 0.10
    return min(1.0, score)


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
    return ValidationResult(
        normalized_text=normalized,
        is_valid=is_valid,
        structure_score=plate_structure_score(normalized),
    )
