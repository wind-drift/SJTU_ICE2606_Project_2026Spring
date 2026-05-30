"""Shared configuration for the deep learning speech pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TRAIN_SPEAKERS = (1, 2, 3, 4, 5)
VAL_SPEAKERS = (6, 7)
DIGIT_LABELS = tuple(range(10))
SNR_LEVELS = (
    -10.0,
    -7.5,
    -5.0,
    -2.5,
    0.0,
    2.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    17.5,
    20.0,
    22.5,
    25.0,
    27.5,
    30.0,
)


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 16_000
    max_seconds: float = 1.5
    preemphasis: float = 0.97
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    n_mfcc: int = 13
    feature_kind: str = "mfcc"
    include_delta: bool = True
    include_delta_delta: bool = True
    denoise: str = "none"
    spectral_subtract_strength: float = 1.0
    spectral_floor: float = 0.05
    fmin: float = 20.0
    fmax: float = 7_600.0

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "FeatureConfig":
        if not values:
            return cls()
        values = dict(values)
        if "feature_kind" not in values:
            values["feature_kind"] = "log_mel"
        if "denoise" not in values:
            values["denoise"] = "none"
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 160
    seed: int = 42
    noise_prob: float = 0.45
    train_noise_kinds: tuple[str, ...] = ("mixed",)
    gain_db: float = 2.0
    time_shift_ms: float = 80.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

