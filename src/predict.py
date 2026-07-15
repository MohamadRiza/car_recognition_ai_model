"""
Vehicle AI - Inference / Prediction Engine
Loads the trained model and produces car attribute predictions from images.
Also generates a rule-based description and price estimate.

Matches the Gemini API output format used in action/cars.js:
{
  "make":         str,   # Not predicted by CNN (needs make dataset)
  "model":        str,   # Not predicted by CNN
  "year":         int,
  "color":        str,
  "price":        str,
  "mileage":      str,
  "bodytype":     str,
  "fueltype":     str,
  "transmission": str,
  "description":  str,
  "confidence":   float
}
"""

import random
from pathlib import Path
from typing import Optional, Dict, Any, Union
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    DEVICE, BODY_TYPES, FUEL_TYPES, TRANSMISSION_TYPES, CAR_COLORS,
    MODEL_CHECKPOINT,
)
from dataset import (
    preprocess_image,
    IDX_TO_BODY_TYPE, IDX_TO_FUEL_TYPE, IDX_TO_TRANS, IDX_TO_COLOR,
)
from model import build_model


# ── Predictor Class ──────────────────────────────────────────

class CarPredictor:
    """
    Inference engine for the trained CarAttributeModel.
    Singleton — load once, predict many times.
    """

    _instance: Optional["CarPredictor"] = None

    def __init__(self, use_checkpoint: bool = True):
        self.model = build_model(pretrained=True, load_checkpoint=use_checkpoint)
        self.model.eval()
        self._model_loaded = MODEL_CHECKPOINT.exists() and use_checkpoint
        print(f"[Predictor] Ready. Using checkpoint: {self._model_loaded}")

    @classmethod
    def get_instance(cls) -> "CarPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.no_grad()
    def predict(self, image: Union[Image.Image, bytes]) -> Dict[str, Any]:
        """
        Predict car attributes from a PIL image or raw bytes.

        Returns a dict matching the Gemini API response schema
        used in the Next.js app (action/cars.js).
        """
        # Accept bytes or PIL
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image or bytes")

        tensor  = preprocess_image(image).to(DEVICE)
        outputs = self.model(tensor)

        # Get predictions + confidence scores
        body_probs  = F.softmax(outputs["body_type"],    dim=1)[0]
        fuel_probs  = F.softmax(outputs["fuel_type"],    dim=1)[0]
        trans_probs = F.softmax(outputs["transmission"], dim=1)[0]
        color_probs = F.softmax(outputs["color"],        dim=1)[0]

        body_idx  = body_probs.argmax().item()
        fuel_idx  = fuel_probs.argmax().item()
        trans_idx = trans_probs.argmax().item()
        color_idx = color_probs.argmax().item()

        body_type    = IDX_TO_BODY_TYPE[body_idx]
        fuel_type    = IDX_TO_FUEL_TYPE[fuel_idx]
        transmission = IDX_TO_TRANS[trans_idx]
        color        = IDX_TO_COLOR[color_idx]

        # Overall confidence = geometric mean of top probabilities
        conf = (
            body_probs[body_idx].item() *
            fuel_probs[fuel_idx].item() *
            trans_probs[trans_idx].item() *
            color_probs[color_idx].item()
        ) ** 0.25

        # Generate rule-based description and estimates
        description = self._generate_description(body_type, fuel_type, transmission, color)
        price       = self._estimate_price(body_type, fuel_type, transmission)
        mileage     = self._estimate_mileage()

        return {
            "make":         "Unknown",          # Requires make-specific training data
            "model":        "Unknown",
            "year":         self._estimate_year(),
            "color":        color,
            "price":        str(price),
            "mileage":      str(mileage),
            "bodytype":     body_type,
            "fueltype":     fuel_type,
            "transmission": transmission,
            "description":  description,
            "confidence":   round(conf, 4),
        }

    # ── Rule-based helpers ────────────────────────────────────

    @staticmethod
    def _estimate_year() -> int:
        """Most cars in listings are recent — estimate between 2018-2024."""
        return random.randint(2018, 2024)

    @staticmethod
    def _estimate_mileage() -> int:
        """Average listing mileage range."""
        return random.randint(5000, 80000)

    @staticmethod
    def _estimate_price(body_type: str, fuel_type: str, transmission: str) -> int:
        base = {
            "SUV": 35000, "Sedan": 25000, "Hatchback": 18000,
            "Convertible": 45000, "Coupe": 38000,
            "Wagon": 28000, "Pickup": 40000,
        }.get(body_type, 25000)

        if fuel_type == "Electric":    base = int(base * 1.3)
        elif fuel_type == "Hybrid":    base = int(base * 1.15)
        if transmission == "Automatic": base = int(base * 1.05)

        # Add variance ±15%
        variance = random.uniform(0.85, 1.15)
        return int(base * variance)

    @staticmethod
    def _generate_description(body_type: str, fuel_type: str, transmission: str, color: str) -> str:
        templates = [
            f"A sleek {color.lower()} {body_type} with a {fuel_type.lower()} engine and {transmission.lower()} "
            f"transmission. This vehicle combines comfort and performance for everyday driving.",

            f"Eye-catching {color.lower()} {body_type} powered by a {fuel_type.lower()} engine. "
            f"Features {transmission.lower()} transmission for a smooth ride. Well-maintained and ready for the road.",

            f"This {color.lower()} {fuel_type.lower()} {body_type} offers {transmission.lower()} transmission "
            f"and excellent build quality. Perfect for both city commuting and highway cruising.",
        ]
        return random.choice(templates)


# ── CLI quick test ────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n--- Prediction Engine Test ---")

    predictor = CarPredictor.get_instance()

    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if not img_path.exists():
            print(f"❌ Image not found: {img_path}")
            sys.exit(1)
        img = Image.open(img_path)
    else:
        # Create a dummy image for testing
        print("No image provided — using dummy 224×224 image")
        img = Image.new("RGB", (224, 224), color=(120, 80, 60))

    result = predictor.predict(img)
    print("\nPrediction Result:")
    for key, value in result.items():
        print(f"  {key:<15}: {value}")

    print(f"\n✅ Confidence: {result['confidence']:.1%}")
