"""Run the full signal-processing plus deep-learning workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import FeatureConfig, SNR_LEVELS, TrainConfig


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis, training, clean eval, and SNR noise eval.")
    parser.add_argument("--reference-dir", default="reference")
    parser.add_argument("--output-dir", default="outputs/deep_learning")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--noise-path", default="reference/mixed_noise.wav")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--feature-kind", choices=["mfcc", "log_mel"], default=FeatureConfig.feature_kind)
    parser.add_argument("--denoise", choices=["none", "notch", "spectral", "notch_spectral"], default=FeatureConfig.denoise)
    parser.add_argument("--n-mels", type=int, default=FeatureConfig.n_mels)
    parser.add_argument("--n-mfcc", type=int, default=FeatureConfig.n_mfcc)
    parser.add_argument("--speaker-id", type=int, default=1)
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--noise-kinds", nargs="+", default=["mixed"])
    parser.add_argument("--snr", nargs="+", type=int, default=list(SNR_LEVELS))
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / "tiny_cnn.pt"

    common_feature_args = [
        "--feature-kind",
        args.feature_kind,
        "--denoise",
        args.denoise,
        "--n-mels",
        str(args.n_mels),
        "--n-mfcc",
        str(args.n_mfcc),
    ]

    _run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.deep_learning.signal_analysis",
            "--reference-dir",
            args.reference_dir,
            "--output-dir",
            str(output_dir / "signal_analysis"),
            "--speaker-id",
            str(args.speaker_id),
            "--digit",
            str(args.digit),
            *common_feature_args,
        ]
    )

    if not args.skip_training:
        _run(
            [
                sys.executable,
                "-B",
                "-m",
                "src.deep_learning.train",
                "--reference-dir",
                args.reference_dir,
                "--output-dir",
                str(output_dir),
                "--checkpoint",
                str(checkpoint),
                "--noise-path",
                args.noise_path,
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                *common_feature_args,
            ]
        )

    _run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.deep_learning.evaluate",
            "--reference-dir",
            args.reference_dir,
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
    )
    _run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.deep_learning.noise_eval",
            "--reference-dir",
            args.reference_dir,
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(checkpoint),
            "--noise-path",
            args.noise_path,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--noise-kinds",
            *args.noise_kinds,
            "--snr",
            *(str(value) for value in args.snr),
        ]
    )
    print(f"Full workflow outputs are under {output_dir}")


if __name__ == "__main__":
    main()
