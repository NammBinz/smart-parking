from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ParkingSession, ParkingSlot, Payment
from app.schemas.dashboard import DashboardResponse


router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(db: Session = Depends(get_db)) -> DashboardResponse:
    start = datetime.combine(datetime.utcnow().date(), datetime.min.time())
    end = start + timedelta(days=1)

    total = db.scalar(select(func.count(ParkingSlot.id))) or 0
    occupied = db.scalar(select(func.count(ParkingSlot.id)).where(ParkingSlot.status == "OCCUPIED")) or 0
    empty = db.scalar(select(func.count(ParkingSlot.id)).where(ParkingSlot.status == "EMPTY")) or 0
    disabled = db.scalar(select(func.count(ParkingSlot.id)).where(ParkingSlot.status == "DISABLED")) or 0
    vehicles_today = db.scalar(
        select(func.count(ParkingSession.id)).where(
            ParkingSession.entry_time >= start,
            ParkingSession.entry_time < end,
        )
    ) or 0
    revenue = db.scalar(
        select(func.coalesce(func.sum(Payment.amount), 0)).where(
            Payment.payment_status == "PAID",
            Payment.paid_at >= start,
            Payment.paid_at < end,
        )
    )
    return DashboardResponse(
        total_slots=total,
        occupied_slots=occupied,
        empty_slots=empty,
        disabled_slots=disabled,
        vehicles_today=vehicles_today,
        revenue_today=Decimal(revenue or 0),
    )
