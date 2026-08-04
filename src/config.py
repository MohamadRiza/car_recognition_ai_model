"""
Vehicle AI - Configuration
Centralised settings for model paths, API config, and inference params.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure dirs exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model Config ────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "efficientnet_b0")   # timm model name
MODEL_CHECKPOINT = MODELS_DIR / "car_model_best.pth"
ONNX_MODEL_PATH = MODELS_DIR / "car_model.onnx"

# Image pre-processing
IMG_SIZE = 224          # EfficientNet-B0 input size
IMG_MEAN = [0.485, 0.456, 0.406]    # ImageNet mean
IMG_STD  = [0.229, 0.224, 0.225]    # ImageNet std
MAX_UPLOAD_MB = 10

# ── Training Config ──────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0         # 0 = main thread (safe on Windows)
DEVICE = "cpu"          # No NVIDIA GPU detected

# ── Car Attributes (classification labels) ──────────────────
CAR_MAKES = ["Toyota", "Honda", "BMW", "Mercedes-Benz", "Tesla", "Ford", "Volkswagen", "Hyundai", "Kia", "Audi"]
BODY_TYPES = ["SUV", "Sedan", "Hatchback", "Convertible", "Coupe", "Wagon", "Pickup"]
FUEL_TYPES = ["Petrol", "Diesel", "Electric", "Hybrid", "Plug-in Hybrid"]
TRANSMISSION_TYPES = ["Automatic", "Manual", "Semi-Automatic"]
CAR_COLORS = [
    "White", "Black", "Silver", "Gray", "Red", "Blue",
    "Green", "Yellow", "Orange", "Brown", "Gold", "Other"
]

# ── API Config ───────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Next.js integration — CORS origin
NEXT_JS_ORIGIN = os.getenv("NEXT_JS_ORIGIN", "http://localhost:3000")
