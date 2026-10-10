import json
import logging
import logging.handlers
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.models import ModelManager
from app.schemas import PredictRequest, PredictResponse


# ============================================================
# Logging Configuration
# ============================================================

os.makedirs("logs", exist_ok=True)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(
                record,
                "%Y-%m-%dT%H:%M:%S"
            ),
            "level": record.levelname,
        }

        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        else:
            payload["message"] = record.getMessage()

        return json.dumps(payload)


logger = logging.getLogger("capstone")
logger.setLevel(logging.INFO)

# Prevent duplicate logs if logging is configured elsewhere
logger.propagate = False

# Rotating JSON file logger
file_handler = logging.handlers.RotatingFileHandler(
    "logs/requests.log",
    maxBytes=5_000_000,
    backupCount=3,
)

file_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)

# Human-readable console logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"
    )
)
logger.addHandler(console_handler)


# ============================================================
# Model Manager
# ============================================================

manager = ModelManager()


# ============================================================
# Prometheus Metrics
# ============================================================

REQUEST_COUNT = Counter(
    "capstone_requests_total",
    "Total prediction requests",
    ["mode", "status"],
)

REQUEST_LATENCY = Histogram(
    "capstone_request_latency_ms",
    "Prediction request latency in milliseconds",
    ["mode"],
    buckets=[
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
    ],
)


def record_metrics(mode, latency_ms, success):
    """
    Record request count and latency metrics for Prometheus.
    """

    REQUEST_COUNT.labels(
        mode=mode,
        status="success" if success else "error",
    ).inc()

    if success:
        REQUEST_LATENCY.labels(
            mode=mode
        ).observe(latency_ms)


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "application_starting",
        extra={
            "extra_fields": {
                "event": "application_starting",
            }
        },
    )

    manager.load_all()

    logger.info(
        "models_loaded",
        extra={
            "extra_fields": {
                "event": "models_loaded",
                "available_modes": manager.available_modes(),
            }
        },
    )

    yield

    logger.info(
        "application_stopping",
        extra={
            "extra_fields": {
                "event": "application_stopping",
            }
        },
    )


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Inference Optimization & Serving Service",
    lifespan=lifespan,
)
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

# ============================================================
# Request Logging Middleware
# ============================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    # Make request ID available to the endpoint if needed
    request.state.request_id = request_id

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "unhandled_error",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                }
            },
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "request_id": request_id,
            },
        )

    latency_ms = (time.time() - start) * 1000

    logger.info(
        "request_completed",
        extra={
            "extra_fields": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
                "mode": getattr(request.state, "mode", None),
            }
        },
    )

    response.headers["X-Request-ID"] = request_id

    return response


# ============================================================
# Validation Error Handler
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
        },
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    modes = manager.available_modes()

    return {
        "status": "ok" if modes else "not_ready",
        "available_modes": modes,
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict", response_model=PredictResponse)
def predict(
    predict_request: PredictRequest,
    request: Request,
):
    start = time.time()

    try:
        label, confidence = manager.predict(
            predict_request.text,
            predict_request.mode,
        )

    except ValueError as e:
        record_metrics(
            predict_request.mode,
            0,
            success=False,
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    latency_ms = (time.time() - start) * 1000

    # Store mode so middleware can access it
    request.state.mode = predict_request.mode

    # Record successful prediction
    record_metrics(
        predict_request.mode,
        latency_ms,
        success=True,
    )

    return PredictResponse(
        label=label,
        confidence=confidence,
        latency_ms=latency_ms,
        mode=predict_request.mode,
    )


# ============================================================
# Prometheus Metrics Endpoint
# ============================================================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )