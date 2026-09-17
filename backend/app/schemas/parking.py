from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field


class CheckInRequest(BaseModel):
    plate_number: str = Field(min_length=1, max_length=30)
    slot_id: int = Field(gt=0)
    entry_image: str | None = Field(default=None, max_length=500)
    entry_detection_confidence: float | None = Field(default=None, ge=0, le=1)
    entry_ocr_confidence: float | None = Field(default=None, ge=0, le=1)


class CheckoutPreviewRequest(BaseModel):
    plate_number: str = Field(min_length=1, max_length=30)


class CheckoutRequest(CheckoutPreviewRequest):
    payment_method: str = Field(min_length=1, max_length=20)


class CheckInResponse(BaseModel):
    message: str
    session_id: int
    plate_number: str
    slot: str
    entry_time: datetime
    status: str


class CheckoutPreviewResponse(BaseModel):
    plate_number: str
    slot: str
    entry_time: datetime
    current_time: datetime
    duration_seconds: int
    billed_hours: int
    price_per_hour: Decimal
    amount: Decimal


class CheckoutResponse(CheckoutPreviewResponse):
    message: str
    payment_id: int
    payment_method: str
    payment_status: str
