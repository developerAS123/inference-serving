from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    # `mode` field gets added here tomorrow (Day 45) once more than one backend exists


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float
    mode: str