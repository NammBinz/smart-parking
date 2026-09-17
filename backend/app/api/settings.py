from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Setting
from app.schemas.settings import SettingsResponse, SettingsUpdate


router = APIRouter(prefix="/settings", tags=["settings"])


def _get_settings(db: Session) -> Setting:
    settings = db.scalar(select(Setting).limit(1))
    if settings is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Settings not configured")
    return settings


@router.get("", response_model=SettingsResponse)
def get_settings(db: Session = Depends(get_db)):
    return _get_settings(db)


@router.patch("", response_model=SettingsResponse)
def update_settings(data: SettingsUpdate, db: Session = Depends(get_db)):
    settings = _get_settings(db)
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(settings, field, value)
    try:
        db.commit()
        db.refresh(settings)
        return settings
    except Exception:
        db.rollback()
        raise
