from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)


class RecognitionDetection(BaseModel):
    class_id: int
    class_name: str
    bbox: BoundingBox
    detection_confidence: float = Field(ge=0, le=1)
    raw_text: str
    normalized_text: str
    ocr_confidence: float = Field(ge=0, le=1)
    preprocessing_variant: str
    is_valid: bool
    ocr_status: str = "recognized"


class RecognitionResponse(BaseModel):
    recognized: bool
    image_path: str
    image_width: int
    image_height: int
    detections: list[RecognitionDetection]
    best_index: int | None
    message: str | None = None


class AIStatusResponse(BaseModel):
    model_available: bool
    model_path: str
    model_names: dict[str, str]
    ocr_available: bool
