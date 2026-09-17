from app.schemas.dashboard import DashboardResponse
from app.schemas.ai import AIStatusResponse, RecognitionResponse
from app.schemas.parking import CheckInRequest, CheckoutPreviewRequest, CheckoutRequest
from app.schemas.settings import SettingsResponse, SettingsUpdate
from app.schemas.slots import SlotCreate, SlotResponse, SlotUpdate

__all__ = [
    "CheckInRequest",
    "AIStatusResponse",
    "CheckoutPreviewRequest",
    "CheckoutRequest",
    "DashboardResponse",
    "SettingsResponse",
    "RecognitionResponse",
    "SettingsUpdate",
    "SlotCreate",
    "SlotResponse",
    "SlotUpdate",
]
