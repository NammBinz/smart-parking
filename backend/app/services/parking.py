import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ParkingSession, ParkingSlot, Payment, Setting
from app.services.plate import normalize_plate


@dataclass(frozen=True)
class FeeCalculation:
    current_time: datetime
    duration_seconds: int
    billed_hours: int
    price_per_hour: Decimal
    amount: Decimal


def _active_session(db: Session, plate_number: str) -> ParkingSession | None:
    return db.scalar(
        select(ParkingSession).where(
            ParkingSession.plate_number == plate_number,
            ParkingSession.status == "PARKING",
        )
    )


def calculate_fee(entry_time: datetime, settings: Setting, now: datetime | None = None) -> FeeCalculation:
    current_time = now or datetime.utcnow()
    duration_seconds = max(0, int((current_time - entry_time).total_seconds()))
    raw_hours = math.ceil(duration_seconds / 3600)
    billed_hours = max(settings.minimum_hours, raw_hours)
    price = Decimal(settings.price_per_hour)
    return FeeCalculation(
        current_time=current_time,
        duration_seconds=duration_seconds,
        billed_hours=billed_hours,
        price_per_hour=price,
        amount=price * billed_hours,
    )


def check_in(db: Session, plate_number: str, slot_id: int) -> tuple[ParkingSession, ParkingSlot]:
    try:
        normalized = normalize_plate(plate_number)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    slot = db.get(ParkingSlot, slot_id)
    if slot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Parking slot not found")
    if not slot.is_active or slot.status == "DISABLED":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Parking slot is disabled")
    if slot.status != "EMPTY":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Parking slot is occupied")
    if _active_session(db, normalized) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Vehicle is already parked")

    parking_session = ParkingSession(
        plate_number=normalized,
        slot_id=slot.id,
        entry_time=datetime.utcnow(),
        status="PARKING",
    )
    slot.status = "OCCUPIED"
    try:
        db.add(parking_session)
        db.commit()
        db.refresh(parking_session)
        return parking_session, slot
    except Exception:
        db.rollback()
        raise


def checkout_preview(db: Session, plate_number: str, now: datetime | None = None):
    try:
        normalized = normalize_plate(plate_number)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    parking_session = _active_session(db, normalized)
    if parking_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active parking session not found")
    settings = db.scalar(select(Setting).limit(1))
    if settings is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Settings not configured")
    return parking_session, calculate_fee(parking_session.entry_time, settings, now)


def checkout(db: Session, plate_number: str, payment_method: str):
    if payment_method not in {"CASH", "TRANSFER"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid payment method")

    parking_session, fee = checkout_preview(db, plate_number)
    slot = parking_session.slot
    if slot is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Parking slot data is missing")

    payment = Payment(
        parking_session_id=parking_session.id,
        billed_hours=fee.billed_hours,
        price_per_hour=fee.price_per_hour,
        amount=fee.amount,
        payment_method=payment_method,
        payment_status="PAID",
        paid_at=fee.current_time,
    )
    parking_session.exit_time = fee.current_time
    parking_session.status = "COMPLETED"
    slot.status = "EMPTY"
    try:
        db.add(payment)
        db.commit()
        db.refresh(payment)
        return parking_session, slot, payment, fee
    except Exception:
        db.rollback()
        raise
