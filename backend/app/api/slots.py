from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.slots import SlotCreate, SlotResponse, SlotUpdate
from app.services import slots as slot_service


router = APIRouter(prefix="/slots", tags=["slots"])


@router.get("", response_model=list[SlotResponse])
def get_slots(db: Session = Depends(get_db)):
    return slot_service.list_slots(db)


@router.post("", response_model=SlotResponse, status_code=status.HTTP_201_CREATED)
def add_slot(data: SlotCreate, db: Session = Depends(get_db)):
    return slot_service.create_slot(db, data)


@router.patch("/{slot_id}", response_model=SlotResponse)
def patch_slot(slot_id: int, data: SlotUpdate, db: Session = Depends(get_db)):
    return slot_service.update_slot(db, slot_id, data)
