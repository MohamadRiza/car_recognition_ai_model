"""
Vehicle AI - Training Script
Fine-tunes EfficientNet-B0 on car images with early stopping,
learning rate scheduling, and checkpoint saving.

Usage:
    python src/train.py
    python src/train.py --epochs 30 --lr 5e-5 --batch 8
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    console = Console()
    RICH = True
except ImportError:
    RICH = False

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_CHECKPOINT, MODELS_DIR, BATCH_SIZE,
)
from dataset import get_train_loader, get_val_loader
from model import build_model, MultiHeadLoss


# ── Training Loop ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiHeadLoss,
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch. Returns (avg_loss, per_head_accuracies)."""
    model.train()
    total_loss = 0.0
    correct: Dict[str, int] = {k: 0 for k in ["make", "body_type", "fuel_type", "transmission", "color"]}
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        targets = {k: v.to(DEVICE) for k, v in labels.items()}

        optimizer.zero_grad()
        outputs = model(images)
        loss, _ = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_size  = images.size(0)
        total_samples += batch_size

        for key in correct:
            preds = outputs[key].argmax(dim=1)
            correct[key] += (preds == targets[key]).sum().item()

    avg_loss  = total_loss / len(loader)
    accuracies = {k: correct[k] / total_samples for k in correct}
    return avg_loss, accuracies


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiHeadLoss,
) -> Tuple[float, Dict[str, float]]:
    """Run one validation epoch. Returns (avg_loss, per_head_accuracies)."""
    model.eval()
    total_loss = 0.0
    correct: Dict[str, int] = {k: 0 for k in ["make", "body_type", "fuel_type", "transmission", "color"]}
    total_samples = 0

    for images, labels in loader:
        images  = images.to(DEVICE)
        targets = {k: v.to(DEVICE) for k, v in labels.items()}

        outputs = model(images)
        loss, _ = criterion(outputs, targets)
        total_loss += loss.item()

        batch_size = images.size(0)
        total_samples += batch_size

        for key in correct:
            preds = outputs[key].argmax(dim=1)
            correct[key] += (preds == targets[key]).sum().item()

    avg_loss   = total_loss / len(loader)
    accuracies = {k: correct[k] / total_samples for k in correct}
    return avg_loss, accuracies


# ── Checkpoint ───────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, best_val_acc, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc":         best_val_acc,
    }, path)
    print(f"  💾 Checkpoint saved → {path}")


# ── Main Training Function ───────────────────────────────────

def train(num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE, batch_size: int = BATCH_SIZE):
    print("\n" + "=" * 60)
    print("  Vehicle AI — Training")
    print(f"  Device  : {DEVICE}")
    print(f"  Model   : EfficientNet-B0 (timm)")
    print(f"  Epochs  : {num_epochs}")
    print(f"  LR      : {lr}")
    print(f"  Batch   : {batch_size}")
    print("=" * 60 + "\n")

    # ── Data ─────────────────────────────────────────────────
    train_loader = get_train_loader()
    val_loader   = get_val_loader()

    if len(train_loader.dataset) == 0:
        print("⚠️  No training data found!")
        print("   Run: python src/dataset.py  for download instructions.")
        print("   Or add images to: data/raw/train/<BodyType>/image.jpg\n")
        return

    # ── Model ─────────────────────────────────────────────────
    model     = build_model(pretrained=True, load_checkpoint=False)
    criterion = MultiHeadLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ── Training Loop ─────────────────────────────────────────
    best_val_acc    = 0.0
    patience        = 5     # Early stopping patience
    patience_counter = 0
    history         = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = validate_epoch(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - start

        # Average accuracy across all heads
        avg_train_acc = sum(train_acc.values()) / len(train_acc)
        avg_val_acc   = sum(val_acc.values()) / len(val_acc)

        print(
            f"Epoch [{epoch:02d}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {avg_train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {avg_val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )
        print(
            f"  make={val_acc['make']:.3f}  "
            f"body_type={val_acc['body_type']:.3f}  "
            f"fuel_type={val_acc['fuel_type']:.3f}  "
            f"transmission={val_acc['transmission']:.3f}  "
            f"color={val_acc['color']:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": avg_train_acc,
            "val_loss": val_loss,     "val_acc": avg_val_acc,
        })

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, MODEL_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                break

    print(f"\n✅ Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Saved to: {MODEL_CHECKPOINT}")
    return history


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Train the Vehicle AI model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr",     type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(num_epochs=args.epochs, lr=args.lr, batch_size=args.batch)
