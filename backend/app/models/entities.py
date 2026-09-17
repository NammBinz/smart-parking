from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def utc_now() -> datetime:
    return datetime.utcnow()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)


class ParkingSlot(TimestampMixin, Base):
    __tablename__ = "parking_slots"
    __table_args__ = (CheckConstraint("status IN ('EMPTY', 'OCCUPIED', 'DISABLED')", name="ck_slot_status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="EMPTY", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    sessions: Mapped[list["ParkingSession"]] = relationship(back_populates="slot")


class ParkingSession(TimestampMixin, Base):
    __tablename__ = "parking_sessions"
    __table_args__ = (
        Index("ix_parking_sessions_plate_status", "plate_number", "status"),
        CheckConstraint("status IN ('PARKING', 'COMPLETED', 'CANCELLED')", name="ck_session_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plate_number: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    slot_id: Mapped[int] = mapped_column(ForeignKey("parking_slots.id"), nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    entry_image: Mapped[str | None] = mapped_column(String(500), nullable=True)
    exit_image: Mapped[str | None] = mapped_column(String(500), nullable=True)
    entry_detection_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="PARKING", nullable=False)

    slot: Mapped[ParkingSlot] = relationship(back_populates="sessions")
    payment: Mapped["Payment | None"] = relationship(back_populates="parking_session", uselist=False)


class Payment(Base):
    __tablename__ = "payments"
    __table_args__ = (
        CheckConstraint("payment_method IN ('CASH', 'TRANSFER')", name="ck_payment_method"),
        CheckConstraint("payment_status IN ('PENDING', 'PAID')", name="ck_payment_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parking_session_id: Mapped[int] = mapped_column(
        ForeignKey("parking_sessions.id"), unique=True, nullable=False
    )
    billed_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    price_per_hour: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    payment_method: Mapped[str] = mapped_column(String(20), nullable=False)
    payment_status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    parking_session: Mapped[ParkingSession] = relationship(back_populates="payment")


class Setting(Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    price_per_hour: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=10000, nullable=False)
    minimum_hours: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    yolo_confidence: Mapped[float] = mapped_column(Float, default=0.25, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="VND", nullable=False)
