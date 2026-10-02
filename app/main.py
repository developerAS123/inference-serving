import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models import ModelManager
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("capstone")

manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.load_all()   
    yield


app = FastAPI(title="Inference Optimization & Serving Service", lifespan=lifespan)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"[{request_id}] unhandled error")
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "request_id": request_id})
    latency_ms = (time.time() - start) * 1000
    logger.info(f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({latency_ms:.1f}ms)")
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": exc.errors()})


@app.get("/health")
def health():
    modes = manager.available_modes()
    return {"status": "ok" if modes else "not_ready", "available_modes": modes}


DEFAULT_MODE = "pytorch-fp32"   # hardcoded today; becomes request.mode tomorrow


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    try:
        label, confidence = manager.predict(request.text, request.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    latency_ms = (time.time() - start) * 1000
    return PredictResponse(label=label, confidence=confidence, latency_ms=latency_ms, mode=request.mode)