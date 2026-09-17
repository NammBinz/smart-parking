from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import Base, SessionLocal, engine
from app.models import ParkingSlot, Setting


DEFAULT_SLOT_CODES = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(1, 11)]


def initialize_database() -> None:
    if engine.url.get_backend_name() == "sqlite":
        database_path = engine.url.database
        if database_path and database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_defaults(db)


def seed_defaults(db: Session) -> None:
    try:
        existing_codes = set(db.scalars(select(ParkingSlot.code)).all())
        for code in DEFAULT_SLOT_CODES:
            if code not in existing_codes:
                db.add(ParkingSlot(code=code, status="EMPTY", is_active=True))

        if db.scalar(select(Setting).limit(1)) is None:
            db.add(
                Setting(
                    price_per_hour=10000,
                    minimum_hours=1,
                    yolo_confidence=0.25,
                    currency="VND",
                )
            )
        db.commit()
    except Exception:
        db.rollback()
        raise
