"""Evaluate validation accuracy under controlled SNR noise."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import SNR_LEVELS, VAL_SPEAKERS
from .dataset import DigitSpeechDataset
from .evaluate import load_checkpoint_model, predict
from .metrics import accuracy, save_accuracy_curve, save_json, save_rows_csv
from .segment import load_digit_samples


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_noise_evaluation(
    checkpoint_path: str | Path = "outputs/deep_learning/tiny_cnn.pt",
    reference_dir: str | Path = "reference",
    output_dir: str | Path = "outputs/deep_learning",
    noise_path: str | Path | None = "reference/mixed_noise.wav",
    noise_kinds: tuple[str, ...] = ("mixed",),
    snr_levels: tuple[float, ...] = SNR_LEVELS,
    batch_size: int = 16,
    seed: int = 42,
    device_name: str = "auto",
) -> list[dict[str, float | str]]:
    device = _device(device_name)
    model, feature_config, _ = load_checkpoint_model(checkpoint_path, device)
    base_samples = load_digit_samples(reference_dir, speaker_ids=VAL_SPEAKERS, config=feature_config)

    rows: list[dict[str, float | str]] = []
    for noise_kind in noise_kinds:
        for snr_db in snr_levels:
            dataset = DigitSpeechDataset(
                samples=base_samples,
                feature_config=feature_config,
                training=False,
                seed=seed + 10_000 + len(rows),
                noise_path=noise_path,
                eval_noise_kind=noise_kind,
                eval_snr_db=float(snr_db),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            y_true, y_pred = predict(model, loader, device)
            acc = accuracy(y_true, y_pred)
            rows.append({"noise_kind": noise_kind, "snr_db": float(snr_db), "accuracy": acc})
            print(f"{noise_kind:>6s} SNR={snr_db:>4} dB accuracy={acc:.3f}")

    output_dir = Path(output_dir)
    save_rows_csv(rows, output_dir / "noise_accuracy.csv")
    save_accuracy_curve(rows, output_dir / "accuracy_snr_curve.png")
    save_json({"rows": rows}, output_dir / "noise_metrics.json")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate noise robustness for TinyKeywordCNN.")
    parser.add_argument("--checkpoint", default="outputs/deep_learning/tiny_cnn.pt")
    parser.add_argument("--reference-dir", default="reference")
    parser.add_argument("--output-dir", default="outputs/deep_learning")
    parser.add_argument("--noise-path", default="reference/mixed_noise.wav")
    parser.add_argument("--noise-kinds", nargs="+", default=["mixed"])
    parser.add_argument("--snr", nargs="+", type=float, default=list(SNR_LEVELS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_noise_evaluation(
        checkpoint_path=args.checkpoint,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        noise_path=args.noise_path,
        noise_kinds=tuple(args.noise_kinds),
        snr_levels=tuple(args.snr),
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
