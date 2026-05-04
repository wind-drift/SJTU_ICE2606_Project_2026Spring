"""PyTorch dataset for segmented digit utterances."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FeatureConfig, TRAIN_SPEAKERS, VAL_SPEAKERS
from .features import (
    add_noise_at_snr,
    extract_features,
    generate_noise,
    load_audio,
    normalize_audio,
)
from .segment import DigitSample, load_digit_samples


class DigitSpeechDataset(Dataset):
    def __init__(
        self,
        reference_dir: str | Path = "reference",
        speaker_ids: Sequence[int] = TRAIN_SPEAKERS,
        feature_config: FeatureConfig | None = None,
        samples: Sequence[DigitSample] | None = None,
        training: bool = False,
        seed: int = 42,
        noise_path: str | Path | None = "reference/mixed_noise.wav",
        noise_prob: float = 0.0,
        noise_snr_range: tuple[float, float] = (0.0, 20.0),
        gain_db: float = 0.0,
        time_shift_ms: float = 0.0,
        eval_noise_kind: str | None = None,
        eval_snr_db: float | None = None,
    ) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.gain_db = gain_db
        self.time_shift_ms = time_shift_ms
        self.eval_noise_kind = eval_noise_kind
        self.eval_snr_db = eval_snr_db

        self.samples = list(samples) if samples is not None else load_digit_samples(
            reference_dir=reference_dir,
            speaker_ids=tuple(speaker_ids),
            config=self.feature_config,
        )
        if not self.samples:
            raise ValueError("DigitSpeechDataset has no samples")

        self.reference_noise: np.ndarray | None = None
        if noise_path is not None and Path(noise_path).exists():
            self.reference_noise, _ = load_audio(noise_path, target_sr=self.feature_config.sr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        audio = sample.waveform.astype(np.float32, copy=True)

        if self.eval_noise_kind is not None and self.eval_snr_db is not None:
            audio = self._apply_fixed_noise(audio, self.eval_noise_kind, self.eval_snr_db)
        elif self.training:
            audio = self._augment(audio)

        features = extract_features(audio, self.feature_config)
        label = np.int64(sample.label)
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32, copy=True)
        if self.gain_db > 0:
            gain = float(self.rng.uniform(-self.gain_db, self.gain_db))
            audio = audio * (10.0 ** (gain / 20.0))

        if self.time_shift_ms > 0:
            max_shift = int(round(self.feature_config.sr * self.time_shift_ms / 1000.0))
            if max_shift > 0:
                shift = int(self.rng.integers(-max_shift, max_shift + 1))
                audio = _time_shift(audio, shift)

        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            snr = float(self.rng.uniform(self.noise_snr_range[0], self.noise_snr_range[1]))
            audio = self._apply_fixed_noise(audio, "mixed", snr)

        return normalize_audio(audio)

    def _apply_fixed_noise(self, audio: np.ndarray, kind: str, snr_db: float) -> np.ndarray:
        noise = generate_noise(
            kind=kind,
            length=len(audio),
            sr=self.feature_config.sr,
            rng=self.rng,
            reference_noise=self.reference_noise,
        )
        return add_noise_at_snr(audio, noise, snr_db)


def _time_shift(audio: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0 or len(audio) == 0:
        return audio
    shifted = np.zeros_like(audio)
    if shift > 0:
        shifted[shift:] = audio[:-shift]
    else:
        shifted[:shift] = audio[-shift:]
    return shifted


def build_train_val_datasets(
    reference_dir: str | Path = "reference",
    feature_config: FeatureConfig | None = None,
    seed: int = 42,
    noise_path: str | Path | None = "reference/mixed_noise.wav",
    noise_prob: float = 0.30,
    gain_db: float = 6.0,
    time_shift_ms: float = 120.0,
) -> tuple[DigitSpeechDataset, DigitSpeechDataset]:
    feature_config = feature_config or FeatureConfig()
    train_ds = DigitSpeechDataset(
        reference_dir=reference_dir,
        speaker_ids=TRAIN_SPEAKERS,
        feature_config=feature_config,
        training=True,
        seed=seed,
        noise_path=noise_path,
        noise_prob=noise_prob,
        noise_snr_range=(0.0, 20.0),
        gain_db=gain_db,
        time_shift_ms=time_shift_ms,
    )
    val_ds = DigitSpeechDataset(
        reference_dir=reference_dir,
        speaker_ids=VAL_SPEAKERS,
        feature_config=feature_config,
        training=False,
        seed=seed + 10_000,
        noise_path=noise_path,
    )
    return train_ds, val_ds

