from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class HistoryItem(BaseModel):
    id: int
    plate_number: str
    slot: str
    entry_time: datetime
    exit_time: datetime | None
    status: str
    billed_hours: int | None
    amount: Decimal | None
    payment_method: str | None
