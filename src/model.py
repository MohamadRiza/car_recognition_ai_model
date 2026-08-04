"""
Vehicle AI - Multi-Head CNN Model
Uses EfficientNet-B0 as backbone (transfer learning from ImageNet).
Multiple classification heads predict different car attributes simultaneously.

Architecture:
    EfficientNet-B0 backbone
        └── Global Average Pool
            ├── head_body_type    (7 classes)
            ├── head_fuel_type    (5 classes)
            ├── head_transmission (3 classes)
            └── head_color        (12 classes)
"""

import torch
import torch.nn as nn
import timm

from config import (
    MODEL_NAME, DEVICE, IMG_SIZE,
    CAR_MAKES, BODY_TYPES, FUEL_TYPES, TRANSMISSION_TYPES, CAR_COLORS,
    MODEL_CHECKPOINT,
)


class CarAttributeModel(nn.Module):
    """
    Multi-head model for simultaneously predicting:
      - Body type   (SUV, Sedan, Hatchback, ...)
      - Fuel type   (Petrol, Diesel, Electric, ...)
      - Transmission (Automatic, Manual, ...)
      - Color       (White, Black, Red, ...)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Backbone: EfficientNet-B0 ─────────────────────────
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,      # Remove default head
            global_pool="avg",  # Global average pooling
        )
        feature_dim = self.backbone.num_features   # 1280 for EfficientNet-B0

        # ── Shared Feature Refinement ─────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        # ── Classification Heads ──────────────────────────────
        self.head_make         = self._make_head(512, len(CAR_MAKES))
        self.head_body_type    = self._make_head(512, len(BODY_TYPES))
        self.head_fuel_type    = self._make_head(512, len(FUEL_TYPES))
        self.head_transmission = self._make_head(512, len(TRANSMISSION_TYPES))
        self.head_color        = self._make_head(512, len(CAR_COLORS))

    @staticmethod
    def _make_head(in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, out_features),
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        shared   = self.shared_fc(features)

        return {
            "make":         self.head_make(shared),
            "body_type":    self.head_body_type(shared),
            "fuel_type":    self.head_fuel_type(shared),
            "transmission": self.head_transmission(shared),
            "color":        self.head_color(shared),
        }


# ── Model Factory ─────────────────────────────────────────────

def build_model(pretrained: bool = True, load_checkpoint: bool = False) -> CarAttributeModel:
    """
    Build and return the CarAttributeModel.

    Args:
        pretrained:       Use ImageNet pretrained weights for backbone.
        load_checkpoint:  Load saved fine-tuned weights if available.
    """
    model = CarAttributeModel(pretrained=pretrained)
    model = model.to(DEVICE)

    if load_checkpoint and MODEL_CHECKPOINT.exists():
        state = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Model] Loaded checkpoint from {MODEL_CHECKPOINT}")
        print(f"        Best Val Accuracy: {state.get('best_val_acc', 'N/A'):.4f}")
    else:
        print(f"[Model] Built fresh model: {MODEL_NAME}")
        print(f"        Pretrained backbone: {pretrained}")
        print(f"        Device: {DEVICE}")

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ── Loss Function ────────────────────────────────────────────

class MultiHeadLoss(nn.Module):
    """
    Weighted cross-entropy loss across all heads.
    Weights let you emphasise more important attributes.
    """

    def __init__(self, weights: dict = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weights = weights or {
            "make":         1.8,
            "body_type":    1.5,
            "fuel_type":    1.0,
            "transmission": 1.0,
            "color":        0.8,
        }

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=DEVICE)
        losses = {}

        for key, weight in self.weights.items():
            loss = self.ce(outputs[key], targets[key])
            losses[key] = loss.item()
            total_loss = total_loss + weight * loss

        return total_loss, losses


# ── Quick sanity check ───────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("\n--- Model Sanity Check ---")
    model = build_model(pretrained=False)

    params = count_parameters(model)
    print(f"\nTotal parameters    : {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    outputs = model(dummy_input)

    print("\nOutput shapes:")
    for k, v in outputs.items():
        print(f"  {k:20s}: {v.shape}")

    print("\n✅ Model architecture verified successfully!")
