"""Train the Tiny CNN digit recognizer."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import FeatureConfig, TrainConfig
from .dataset import build_train_val_datasets
from .metrics import save_json
from .model import TinyKeywordCNN, count_parameters
from .segment import write_manifest


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyKeywordCNN for isolated digit recognition.")
    parser.add_argument("--reference-dir", default="reference")
    parser.add_argument("--output-dir", default="outputs/deep_learning")
    parser.add_argument("--checkpoint", default="outputs/deep_learning/tiny_cnn.pt")
    parser.add_argument("--noise-path", default="reference/mixed_noise.wav")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--noise-prob", type=float, default=TrainConfig.noise_prob)
    parser.add_argument("--gain-db", type=float, default=TrainConfig.gain_db)
    parser.add_argument("--time-shift-ms", type=float, default=TrainConfig.time_shift_ms)
    parser.add_argument("--max-seconds", type=float, default=FeatureConfig.max_seconds)
    parser.add_argument("--feature-kind", choices=["mfcc", "log_mel"], default=FeatureConfig.feature_kind)
    parser.add_argument("--denoise", choices=["none", "notch", "spectral", "notch_spectral"], default=FeatureConfig.denoise)
    parser.add_argument("--n-mels", type=int, default=FeatureConfig.n_mels)
    parser.add_argument("--n-mfcc", type=int, default=FeatureConfig.n_mfcc)
    parser.add_argument("--no-delta", action="store_true")
    parser.add_argument("--no-delta-delta", action="store_true")
    parser.add_argument("--spectral-subtract-strength", type=float, default=FeatureConfig.spectral_subtract_strength)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    feature_config = FeatureConfig(
        max_seconds=args.max_seconds,
        feature_kind=args.feature_kind,
        denoise=args.denoise,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        include_delta=not args.no_delta,
        include_delta_delta=not args.no_delta and not args.no_delta_delta,
        spectral_subtract_strength=args.spectral_subtract_strength,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        noise_prob=args.noise_prob,
        gain_db=args.gain_db,
        time_shift_ms=args.time_shift_ms,
    )

    train_ds, val_ds = build_train_val_datasets(
        reference_dir=args.reference_dir,
        feature_config=feature_config,
        seed=args.seed,
        noise_path=args.noise_path,
        noise_prob=args.noise_prob,
        gain_db=args.gain_db,
        time_shift_ms=args.time_shift_ms,
    )
    write_manifest(train_ds.samples + val_ds.samples, output_dir / "segments_manifest.json")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    model = TinyKeywordCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Training samples: {len(train_ds)}, validation samples: {len(val_ds)}, device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_config": feature_config.to_dict(),
                    "train_config": train_config.to_dict(),
                    "parameter_count": count_parameters(model),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_json(
        {
            "feature_config": feature_config.to_dict(),
            "train_config": train_config.to_dict(),
            "parameter_count": count_parameters(model),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "checkpoint": checkpoint_path.as_posix(),
        },
        output_dir / "train_summary.json",
    )
    print(f"Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()

