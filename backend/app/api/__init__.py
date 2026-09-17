from fastapi import APIRouter

from app.api import ai, dashboard, history, parking, settings, slots

api_router = APIRouter(prefix="/api")
api_router.include_router(ai.router)
api_router.include_router(dashboard.router)
api_router.include_router(slots.router)
api_router.include_router(parking.router)
api_router.include_router(history.router)
api_router.include_router(settings.router)

__all__ = ["api_router"]
