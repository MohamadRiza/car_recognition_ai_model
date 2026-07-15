"""
Vehicle AI - Dataset & Data Pipeline
Handles loading, transforming, and preparing car images for training.

Supported datasets:
  - Stanford Cars Dataset  (196 makes/models, 16,185 images)
  - Custom dataset         (your own car images in data/raw/)
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    IMG_SIZE, IMG_MEAN, IMG_STD,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    BODY_TYPES, FUEL_TYPES, TRANSMISSION_TYPES, CAR_COLORS,
    BATCH_SIZE, NUM_WORKERS,
)


# ── Transforms ───────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Augmented transforms for training to improve generalisation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Clean transforms for validation / inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


# ── Label Maps ───────────────────────────────────────────────

BODY_TYPE_TO_IDX  = {b: i for i, b in enumerate(BODY_TYPES)}
FUEL_TYPE_TO_IDX  = {f: i for i, f in enumerate(FUEL_TYPES)}
TRANS_TO_IDX      = {t: i for i, t in enumerate(TRANSMISSION_TYPES)}
COLOR_TO_IDX      = {c: i for i, c in enumerate(CAR_COLORS)}

IDX_TO_BODY_TYPE  = {v: k for k, v in BODY_TYPE_TO_IDX.items()}
IDX_TO_FUEL_TYPE  = {v: k for k, v in FUEL_TYPE_TO_IDX.items()}
IDX_TO_TRANS      = {v: k for k, v in TRANS_TO_IDX.items()}
IDX_TO_COLOR      = {v: k for k, v in COLOR_TO_IDX.items()}


# ── Custom Dataset ───────────────────────────────────────────

class CarDataset(Dataset):
    """
    Car image dataset.

    Expected folder structure:
        data/raw/
          train/
            SUV/
              car1.jpg
              car2.jpg
            Sedan/
              car3.jpg
          val/
            SUV/
              car4.jpg

    Or provide a JSON annotations file with format:
        [
          {
            "image_path": "data/raw/train/SUV/car1.jpg",
            "body_type":  "SUV",
            "color":      "White",
            "fuel_type":  "Petrol",
            "transmission": "Automatic"
          },
          ...
        ]
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",          # "train" | "val" | "test"
        annotations_file: Optional[Path] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or get_val_transforms()
        self.samples: List[Dict] = []

        if annotations_file and annotations_file.exists():
            self._load_from_json(annotations_file)
        else:
            self._load_from_folders()

    def _load_from_folders(self):
        """Auto-discover images from class-named subdirectories."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            print(f"[Dataset] '{split_dir}' not found — using empty dataset.")
            return

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append({
                        "image_path": str(img_file),
                        "body_type": class_name if class_name in BODY_TYPES else "SUV",
                        "color": "White",
                        "fuel_type": "Petrol",
                        "transmission": "Automatic",
                    })

    def _load_from_json(self, annotations_file: Path):
        """Load samples from a JSON annotations file."""
        with open(annotations_file, "r") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Build label tensors
        labels = {
            "body_type":    torch.tensor(BODY_TYPE_TO_IDX.get(sample["body_type"], 0), dtype=torch.long),
            "fuel_type":    torch.tensor(FUEL_TYPE_TO_IDX.get(sample["fuel_type"], 0), dtype=torch.long),
            "transmission": torch.tensor(TRANS_TO_IDX.get(sample["transmission"], 0), dtype=torch.long),
            "color":        torch.tensor(COLOR_TO_IDX.get(sample["color"], 11), dtype=torch.long),
        }

        return img, labels


# ── DataLoader Factories ──────────────────────────────────────

def get_train_loader(root_dir: Path = RAW_DATA_DIR) -> DataLoader:
    dataset = CarDataset(root_dir, split="train", transform=get_train_transforms())
    print(f"[Dataset] Training samples: {len(dataset)}")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


def get_val_loader(root_dir: Path = RAW_DATA_DIR) -> DataLoader:
    dataset = CarDataset(root_dir, split="val", transform=get_val_transforms())
    print(f"[Dataset] Validation samples: {len(dataset)}")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# ── Image Pre-processing Utility (for inference) ─────────────

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.
    Returns a (1, 3, H, W) tensor.
    """
    transform = get_val_transforms()
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0)  # Add batch dimension


# ── Stanford Cars Dataset Download Helper ────────────────────

def download_stanford_cars(dest: Path = RAW_DATA_DIR):
    """
    Instructions to manually download the Stanford Cars Dataset.
    (Automated download requires kaggle API credentials)
    """
    print("\n" + "=" * 60)
    print("Stanford Cars Dataset Setup")
    print("=" * 60)
    print(
        "\nOption 1 — Kaggle CLI (recommended):\n"
        "  pip install kaggle\n"
        "  # Place kaggle.json in ~/.kaggle/\n"
        "  kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars\n"
        f"  # Unzip to: {dest}\n"
    )
    print(
        "Option 2 — Direct download:\n"
        "  https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars\n"
    )
    print(
        "Expected folder structure after unzip:\n"
        f"  {dest}/train/  (class folders)\n"
        f"  {dest}/val/    (class folders)\n"
    )


if __name__ == "__main__":
    download_stanford_cars()
