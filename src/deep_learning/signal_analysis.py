"""Generate signal-analysis figures required by the speech-recognition task."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .config import DEFAULT_DATA_DIR, FeatureConfig
from .features import (
    log_mel_matrix,
    mfcc_spectrogram,
    normalize_audio,
    prepare_waveform,
    rms,
)
from .metrics import save_json
from .segment import load_digit_samples


def _save_waveform(audio: np.ndarray, sr: int, path: Path) -> None:
    times = np.arange(len(audio), dtype=np.float32) / float(sr)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, audio, linewidth=0.8)
    ax.set_title("Time-domain waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_fft(audio: np.ndarray, sr: int, path: Path) -> float:
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    magnitude = np.abs(spectrum)
    dominant_idx = int(np.argmax(magnitude[1:]) + 1) if len(magnitude) > 1 else 0

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, 20.0 * np.log10(magnitude + 1e-8), linewidth=0.8)
    ax.set_title("FFT magnitude spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, min(sr / 2, 8000))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return float(freqs[dominant_idx])


def _save_stft(audio: np.ndarray, config: FeatureConfig, path: Path) -> None:
    freqs, times, stft = signal.spectrogram(
        audio,
        fs=config.sr,
        window="hann",
        nperseg=config.win_length,
        noverlap=max(0, config.win_length - config.hop_length),
        nfft=config.n_fft,
        mode="magnitude",
    )
    power_db = 20.0 * np.log10(stft + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.pcolormesh(times, freqs, power_db, shading="auto", cmap="magma")
    ax.set_title("STFT spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, min(config.fmax, config.sr / 2))
    fig.colorbar(image, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_feature_map(features: np.ndarray, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(features, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_signal_analysis(
    reference_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = "outputs/deep_learning/signal_analysis",
    speaker_id: int = 1,
    digit: int = 0,
    feature_config: FeatureConfig | None = None,
) -> dict[str, object]:
    feature_config = feature_config or FeatureConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_digit_samples(reference_dir, speaker_ids=(speaker_id,), config=feature_config)
    matches = [sample for sample in samples if sample.label == digit]
    if not matches:
        raise ValueError(f"No digit {digit} sample found for speaker {speaker_id}")

    sample = matches[0]
    raw_audio = normalize_audio(sample.waveform)
    processed_audio = prepare_waveform(raw_audio, feature_config)
    dominant_freq = _save_fft(raw_audio, feature_config.sr, output_dir / "fft_magnitude.png")
    _save_waveform(raw_audio, feature_config.sr, output_dir / "time_waveform.png")
    _save_stft(processed_audio, feature_config, output_dir / "stft_spectrogram.png")

    log_mel = log_mel_matrix(raw_audio, feature_config)
    mfcc = mfcc_spectrogram(raw_audio, feature_config)[0]
    _save_feature_map(log_mel, output_dir / "log_mel_features.png", "Log-Mel filter-bank features", "Mel bin")
    _save_feature_map(mfcc, output_dir / "mfcc_features.png", "MFCC / delta feature map", "Coefficient")

    summary: dict[str, object] = {
        "source": sample.metadata.to_dict(),
        "signal_processing": {
            "sample_rate_hz": feature_config.sr,
            "preemphasis": feature_config.preemphasis,
            "frame_length_samples": feature_config.win_length,
            "hop_length_samples": feature_config.hop_length,
            "n_fft": feature_config.n_fft,
            "n_mels": feature_config.n_mels,
            "n_mfcc": feature_config.n_mfcc,
            "feature_kind": feature_config.feature_kind,
            "include_delta": feature_config.include_delta,
            "include_delta_delta": feature_config.include_delta_delta,
            "denoise": feature_config.denoise,
        },
        "measurements": {
            "duration_sec": len(raw_audio) / float(feature_config.sr),
            "peak": float(np.max(np.abs(raw_audio))) if len(raw_audio) else 0.0,
            "rms": rms(raw_audio),
            "dominant_fft_frequency_hz": dominant_freq,
            "log_mel_shape": list(log_mel.shape),
            "mfcc_shape": list(mfcc.shape),
        },
        "figures": {
            "time_waveform": "time_waveform.png",
            "fft_magnitude": "fft_magnitude.png",
            "stft_spectrogram": "stft_spectrogram.png",
            "log_mel_features": "log_mel_features.png",
            "mfcc_features": "mfcc_features.png",
        },
    }
    save_json(summary, output_dir / "signal_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create signal-analysis figures for one digit utterance.")
    parser.add_argument("--reference-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="outputs/deep_learning/signal_analysis")
    parser.add_argument("--speaker-id", type=int, default=1)
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--feature-kind", choices=["mfcc", "log_mel"], default=FeatureConfig.feature_kind)
    parser.add_argument("--denoise", choices=["none", "notch", "spectral", "notch_spectral"], default=FeatureConfig.denoise)
    parser.add_argument("--n-mels", type=int, default=FeatureConfig.n_mels)
    parser.add_argument("--n-mfcc", type=int, default=FeatureConfig.n_mfcc)
    parser.add_argument("--no-delta", action="store_true")
    parser.add_argument("--no-delta-delta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatureConfig(
        feature_kind=args.feature_kind,
        denoise=args.denoise,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        include_delta=not args.no_delta,
        include_delta_delta=not args.no_delta and not args.no_delta_delta,
    )
    summary = run_signal_analysis(
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        speaker_id=args.speaker_id,
        digit=args.digit,
        feature_config=config,
    )
    print(f"Wrote signal analysis to {args.output_dir}")
    print(f"Selected source: {summary['source']}")


if __name__ == "__main__":
    main()
