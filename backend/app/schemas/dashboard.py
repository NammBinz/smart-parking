from decimal import Decimal

from pydantic import BaseModel


class DashboardResponse(BaseModel):
    total_slots: int
    occupied_slots: int
    empty_slots: int
    disabled_slots: int
    vehicles_today: int
    revenue_today: Decimal
