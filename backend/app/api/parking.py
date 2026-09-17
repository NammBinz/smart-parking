from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.parking import (
    CheckInRequest,
    CheckInResponse,
    CheckoutPreviewRequest,
    CheckoutPreviewResponse,
    CheckoutRequest,
    CheckoutResponse,
)
from app.services import parking as parking_service


router = APIRouter(prefix="/parking", tags=["parking"])


@router.post("/check-in", response_model=CheckInResponse, status_code=status.HTTP_201_CREATED)
def check_in(data: CheckInRequest, db: Session = Depends(get_db)):
    parking_session, slot = parking_service.check_in(db, data.plate_number, data.slot_id)
    return CheckInResponse(
        message="Vehicle checked in successfully",
        session_id=parking_session.id,
        plate_number=parking_session.plate_number,
        slot=slot.code,
        entry_time=parking_session.entry_time,
        status=parking_session.status,
    )


@router.post("/checkout-preview", response_model=CheckoutPreviewResponse)
def preview_checkout(data: CheckoutPreviewRequest, db: Session = Depends(get_db)):
    parking_session, fee = parking_service.checkout_preview(db, data.plate_number)
    return CheckoutPreviewResponse(
        plate_number=parking_session.plate_number,
        slot=parking_session.slot.code,
        entry_time=parking_session.entry_time,
        **fee.__dict__,
    )


@router.post("/checkout", response_model=CheckoutResponse)
def confirm_checkout(data: CheckoutRequest, db: Session = Depends(get_db)):
    parking_session, slot, payment, fee = parking_service.checkout(
        db, data.plate_number, data.payment_method
    )
    return CheckoutResponse(
        message="Payment completed and vehicle checked out",
        payment_id=payment.id,
        payment_method=payment.payment_method,
        payment_status=payment.payment_status,
        plate_number=parking_session.plate_number,
        slot=slot.code,
        entry_time=parking_session.entry_time,
        **fee.__dict__,
    )
