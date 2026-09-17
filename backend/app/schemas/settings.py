from decimal import Decimal

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


class SettingsResponse(ORMModel):
    id: int
    price_per_hour: Decimal
    minimum_hours: int
    yolo_confidence: float
    currency: str


class SettingsUpdate(BaseModel):
    price_per_hour: Decimal | None = Field(default=None, ge=0)
    minimum_hours: int | None = Field(default=None, ge=1)
    yolo_confidence: float | None = Field(default=None, ge=0, le=1)
    currency: str | None = Field(default=None, min_length=1, max_length=10)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Currency is required")
        return normalized
