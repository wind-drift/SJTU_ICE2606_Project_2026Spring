"""Evaluate the clean validation set and save a confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import FeatureConfig, VAL_SPEAKERS
from .dataset import DigitSpeechDataset
from .metrics import (
    accuracy,
    confusion_matrix,
    plot_confusion_matrix,
    save_confusion_csv,
    save_json,
)
from .model import TinyKeywordCNN


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[TinyKeywordCNN, FeatureConfig, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run python -m src.deep_learning.train first.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_config = FeatureConfig.from_dict(checkpoint.get("feature_config"))
    model = TinyKeywordCNN(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, feature_config, checkpoint


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    for features, labels in loader:
        logits = model(features.to(device))
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(int(pred) for pred in preds)
        y_true.extend(int(label) for label in labels.tolist())
    return y_true, y_pred


def evaluate_clean(
    checkpoint_path: str | Path = "outputs/deep_learning/tiny_cnn.pt",
    reference_dir: str | Path = "reference",
    output_dir: str | Path = "outputs/deep_learning",
    batch_size: int = 16,
    device_name: str = "auto",
) -> dict[str, float | int | str]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)
    model, feature_config, checkpoint = load_checkpoint_model(checkpoint_path, device)
    dataset = DigitSpeechDataset(
        reference_dir=reference_dir,
        speaker_ids=VAL_SPEAKERS,
        feature_config=feature_config,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred = predict(model, loader, device)
    acc = accuracy(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, num_classes=10)

    output_dir = Path(output_dir)
    save_confusion_csv(matrix, output_dir / "clean_confusion_matrix.csv")
    plot_confusion_matrix(matrix, output_dir / "clean_confusion_matrix.png", title="Clean validation confusion matrix")
    metrics = {
        "accuracy": acc,
        "num_samples": len(y_true),
        "checkpoint": Path(checkpoint_path).as_posix(),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "best_val_accuracy_recorded": float(checkpoint.get("best_val_accuracy", 0.0)),
    }
    save_json(metrics, output_dir / "clean_metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TinyKeywordCNN on clean validation speakers.")
    parser.add_argument("--checkpoint", default="outputs/deep_learning/tiny_cnn.pt")
    parser.add_argument("--reference-dir", default="reference")
    parser.add_argument("--output-dir", default="outputs/deep_learning")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_clean(
        checkpoint_path=args.checkpoint,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(f"Clean accuracy: {metrics['accuracy']:.3f} over {metrics['num_samples']} samples")


if __name__ == "__main__":
    main()
