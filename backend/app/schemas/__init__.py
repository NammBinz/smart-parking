from app.schemas.dashboard import DashboardResponse
from app.schemas.parking import CheckInRequest, CheckoutPreviewRequest, CheckoutRequest
from app.schemas.settings import SettingsResponse, SettingsUpdate
from app.schemas.slots import SlotCreate, SlotResponse, SlotUpdate

__all__ = [
    "CheckInRequest",
    "CheckoutPreviewRequest",
    "CheckoutRequest",
    "DashboardResponse",
    "SettingsResponse",
    "SettingsUpdate",
    "SlotCreate",
    "SlotResponse",
    "SlotUpdate",
]
