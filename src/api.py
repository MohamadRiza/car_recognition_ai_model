"""
Vehicle AI - FastAPI Server
Serves the trained car prediction model as a REST API.
The Next.js app sends car images here instead of calling Gemini.

Endpoints:
  GET  /           → Welcome message
  GET  /health     → Health check
  POST /predict    → Car image → prediction JSON

Usage:
  python src/api.py
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from loguru import logger

from config import API_HOST, API_PORT, API_RELOAD, NEXT_JS_ORIGIN, MAX_UPLOAD_MB


# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(
    title="Vehicle AI API",
    description="Custom car image analysis API — Drop-in replacement for Google Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Next.js dev server & all local origin ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy model loading (loads on first request, not on startup) ─

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        logger.info("Loading model for the first time...")
        from predict import CarPredictor
        _predictor = CarPredictor.get_instance()
        logger.success("Model loaded and ready!")
    return _predictor


# ── Request / Response Schemas ───────────────────────────────

class PredictionResult(BaseModel):
    make:         str
    model:        str
    year:         int
    color:        str
    price:        str
    mileage:      str
    bodytype:     str
    fueltype:     str
    transmission: str
    description:  str
    confidence:   float


class PredictionResponse(BaseModel):
    success:   bool
    data:      Optional[PredictionResult] = None
    error:     Optional[str] = None
    took_ms:   Optional[float] = None
    source:    str = "car_recognition_ai_model"


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str = "1.0.0"
    engine:       str = "PyTorch Vision Transformer & EfficientNet"


# ── Middleware: Request Logging ───────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response


# ── Routes ───────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    return {
        "name":    "Vehicle AI API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
        version="1.0.0",
        engine="PyTorch Vision Transformer & EfficientNet",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_car(file: UploadFile = File(...)):
    """
    Analyse a car image and return structured car details.

    Accepts: JPEG, PNG, WebP (max 10 MB)
    Returns: JSON with make, model, year, color, body type, fuel type,
             transmission, mileage, price, description, confidence
    """

    # ── Read & validate size ──────────────────────────────────
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_UPLOAD_MB} MB."
        )

    # ── Load image ────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read or parse image file.")

    # ── Run prediction ────────────────────────────────────────
    try:
        start = time.time()
        predictor = get_predictor()
        result    = predictor.predict(image, filename=file.filename)
        elapsed   = (time.time() - start) * 1000

        logger.info(
            f"Predicted: {result.get('make')} {result.get('model')} ({result.get('year')}) | "
            f"{result.get('bodytype')} | {result.get('color')} | "
            f"conf={result.get('confidence', 0):.2%} | {elapsed:.0f}ms"
        )

        return PredictionResponse(
            success=True,
            data=PredictionResult(**result),
            took_ms=round(elapsed, 1),
            source="car_recognition_ai_model",
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
        return PredictionResponse(
            success=False,
            error=str(e),
            took_ms=0.0,
            source="car_recognition_ai_model",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


# ── Dev Server Entry Point ────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Vehicle AI API on http://{API_HOST}:{API_PORT}")
    logger.info(f"Docs: http://localhost:{API_PORT}/docs")

    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
