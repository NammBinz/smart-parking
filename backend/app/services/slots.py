from fastapi import HTTPException, status
import re

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import ParkingSession, ParkingSlot
from app.schemas.slots import SlotCreate, SlotResponse, SlotUpdate


def list_slots(db: Session) -> list[SlotResponse]:
    slots = db.scalars(select(ParkingSlot)).all()
    slots.sort(key=lambda slot: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", slot.code)])
    active_by_slot = {
        session.slot_id: session
        for session in db.scalars(
            select(ParkingSession).where(ParkingSession.status == "PARKING")
        ).all()
    }
    return [
        SlotResponse(
            id=slot.id,
            code=slot.code,
            status=slot.status,
            is_active=slot.is_active,
            created_at=slot.created_at,
            updated_at=slot.updated_at,
            current_parking=(
                {
                    "plate_number": active_by_slot[slot.id].plate_number,
                    "entry_time": active_by_slot[slot.id].entry_time,
                }
                if slot.id in active_by_slot
                else None
            ),
        )
        for slot in slots
    ]


def create_slot(db: Session, data: SlotCreate) -> ParkingSlot:
    slot = ParkingSlot(code=data.code, status="EMPTY", is_active=True)
    try:
        db.add(slot)
        db.commit()
        db.refresh(slot)
        return slot
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Slot code already exists") from exc


def update_slot(db: Session, slot_id: int, data: SlotUpdate) -> ParkingSlot:
    slot = db.get(ParkingSlot, slot_id)
    if slot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Parking slot not found")

    changes = data.model_dump(exclude_unset=True)
    requested_active = changes.get("is_active")
    requested_status = changes.get("status")
    if (requested_active is False and requested_status == "EMPTY") or (
        requested_active is True and requested_status == "DISABLED"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="is_active and status values are inconsistent",
        )
    if slot.status == "OCCUPIED" and (requested_active is False or requested_status in {"EMPTY", "DISABLED"}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An occupied slot cannot be disabled or emptied")
    if requested_status == "OCCUPIED" and slot.status != "OCCUPIED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Use check-in to occupy a slot")

    if "code" in changes:
        slot.code = changes["code"]
    if requested_active is False:
        slot.is_active = False
        slot.status = "DISABLED"
    elif requested_active is True:
        slot.is_active = True
        if slot.status == "DISABLED":
            slot.status = "EMPTY"
    if requested_status == "DISABLED":
        slot.is_active = False
        slot.status = "DISABLED"
    elif requested_status == "EMPTY":
        slot.is_active = True
        slot.status = "EMPTY"

    try:
        db.commit()
        db.refresh(slot)
        return slot
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Slot code already exists") from exc
