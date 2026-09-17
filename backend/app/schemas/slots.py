from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


SlotStatus = Literal["EMPTY", "OCCUPIED", "DISABLED"]


def normalize_slot_code(value: str) -> str:
    value = value.strip().upper()
    if not value:
        raise ValueError("Slot code is required")
    return value


class SlotCreate(BaseModel):
    code: str = Field(max_length=20)

    _normalize_code = field_validator("code")(normalize_slot_code)


class SlotUpdate(BaseModel):
    code: str | None = Field(default=None, max_length=20)
    is_active: bool | None = None
    status: SlotStatus | None = None

    _normalize_code = field_validator("code")(normalize_slot_code)


class CurrentParkingInfo(BaseModel):
    plate_number: str
    entry_time: datetime


class SlotResponse(ORMModel):
    id: int
    code: str
    status: SlotStatus
    is_active: bool
    created_at: datetime
    updated_at: datetime
    current_parking: CurrentParkingInfo | None = None
