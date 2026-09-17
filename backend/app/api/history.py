from datetime import date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ParkingSession, ParkingSlot, Payment
from app.schemas.history import HistoryItem
from app.services.plate import normalize_plate


router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[HistoryItem])
def get_history(
    plate_number: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    db: Session = Depends(get_db),
):
    statement = (
        select(ParkingSession, ParkingSlot, Payment)
        .join(ParkingSlot, ParkingSession.slot_id == ParkingSlot.id)
        .outerjoin(Payment, Payment.parking_session_id == ParkingSession.id)
        .order_by(ParkingSession.entry_time.desc())
    )
    if plate_number:
        try:
            normalized = normalize_plate(plate_number)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        statement = statement.where(ParkingSession.plate_number == normalized)
    if date_from:
        statement = statement.where(ParkingSession.entry_time >= datetime.combine(date_from, time.min))
    if date_to:
        statement = statement.where(
            ParkingSession.entry_time < datetime.combine(date_to, time.min) + timedelta(days=1)
        )

    return [
        HistoryItem(
            id=parking_session.id,
            plate_number=parking_session.plate_number,
            slot=slot.code,
            entry_time=parking_session.entry_time,
            exit_time=parking_session.exit_time,
            status=parking_session.status,
            billed_hours=payment.billed_hours if payment else None,
            amount=payment.amount if payment else None,
            payment_method=payment.payment_method if payment else None,
        )
        for parking_session, slot, payment in db.execute(statement).all()
    ]
