from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["pytorch-fp32", "onnx-fp32", "onnx-int8"] = Field(
        default="pytorch-fp32",
        description="Which backend variant to run inference with",
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float
    mode: str