"""Audio loading, feature extraction, and noise utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

from .config import FeatureConfig


EPS = 1e-8


def load_audio(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load mono audio as float32.

    MP3 decoding depends on optional runtime packages. The function tries
    librosa, soundfile, and scipy wavfile in that order and reports all
    backend errors if none can read the file.
    """

    path = Path(path)
    errors: list[str] = []

    try:
        import librosa  # type: ignore

        audio, sr = librosa.load(path.as_posix(), sr=target_sr, mono=True)
        return normalize_audio(audio.astype(np.float32)), int(target_sr)
    except Exception as exc:  # pragma: no cover - backend dependent
        errors.append(f"librosa: {exc}")

    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(path.as_posix(), always_2d=False)
        audio = _mono_float32(audio)
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr
        return normalize_audio(audio), int(sr)
    except Exception as exc:  # pragma: no cover - backend dependent
        errors.append(f"soundfile: {exc}")

    if path.suffix.lower() == ".wav":
        try:
            sr, audio = wavfile.read(path.as_posix())
            audio = _mono_float32(audio)
            if sr != target_sr:
                audio = resample_audio(audio, sr, target_sr)
                sr = target_sr
            return normalize_audio(audio), int(sr)
        except Exception as exc:
            errors.append(f"scipy.wavfile: {exc}")

    message = "\n  - ".join(errors)
    raise RuntimeError(
        f"Could not decode audio file {path}. Install requirements.txt and make "
        f"sure MP3 decoding is available.\n  - {message}"
    )


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        audio = audio.astype(np.float32) / float(scale)
    else:
        audio = audio.astype(np.float32)
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, copy=False)
    if audio.size == 0:
        return audio
    audio = audio - float(audio.mean())
    peak = float(np.max(np.abs(audio)))
    if peak > EPS:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    gcd = math.gcd(int(orig_sr), int(target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32)


def pad_or_truncate(audio: np.ndarray, target_len: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        return audio[start : start + target_len]
    pad_total = target_len - len(audio)
    left = pad_total // 2
    right = pad_total - left
    return np.pad(audio, (left, right), mode="constant").astype(np.float32)


def pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if len(audio) <= 1 or coeff <= 0:
        return audio.astype(np.float32, copy=False)
    out = np.empty_like(audio, dtype=np.float32)
    out[0] = audio[0]
    out[1:] = audio[1:] - coeff * audio[:-1]
    return out


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def mel_to_hz(freq_mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(freq_mel) / 2595.0) - 1.0)


def mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None = None,
) -> np.ndarray:
    fmax = float(fmax or sr / 2)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for idx in range(n_mels):
        left, center, right = bins[idx : idx + 3]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        center = min(center, filters.shape[1] - 1)
        right = min(right, filters.shape[1])
        if center > left:
            filters[idx, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        if right > center:
            filters[idx, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)

    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], EPS)
    filters *= enorm[:, np.newaxis].astype(np.float32)
    return filters


def log_mel_spectrogram(audio: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Convert one utterance to a normalized 1 x n_mels x frames tensor."""

    target_len = int(round(config.sr * config.max_seconds))
    audio = pad_or_truncate(normalize_audio(audio), target_len)
    audio = pre_emphasis(audio, config.preemphasis)

    waveform = torch.from_numpy(audio.astype(np.float32))
    window = torch.hann_window(config.win_length)
    spec = torch.stft(
        waveform,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2)
    filters = torch.from_numpy(
        mel_filter_bank(config.sr, config.n_fft, config.n_mels, config.fmin, config.fmax)
    )
    mel = torch.matmul(filters, power).clamp_min(EPS)
    log_mel = torch.log(mel)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std(unbiased=False) + 1e-5)
    return log_mel.unsqueeze(0).numpy().astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + EPS))


def fit_noise_length(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(noise) < length:
        repeats = int(math.ceil(length / len(noise)))
        noise = np.tile(noise, repeats)
    if len(noise) > length:
        start = int(rng.integers(0, len(noise) - length + 1))
        noise = noise[start : start + length]
    return normalize_audio(noise)


def generate_noise(
    kind: str,
    length: int,
    sr: int,
    rng: np.random.Generator,
    reference_noise: np.ndarray | None = None,
) -> np.ndarray:
    kind = kind.lower()
    if kind == "mixed" and reference_noise is not None:
        return fit_noise_length(reference_noise, length, rng)

    if kind == "white":
        noise = rng.standard_normal(length)
    elif kind == "pink":
        freqs = np.fft.rfftfreq(length, d=1.0 / sr)
        weights = np.ones_like(freqs)
        weights[1:] = 1.0 / np.sqrt(np.maximum(freqs[1:], EPS))
        spectrum = (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))) * weights
        spectrum[0] = 0.0
        if length % 2 == 0:
            spectrum[-1] = spectrum[-1].real + 0j
        noise = np.fft.irfft(spectrum, n=length)
    elif kind in {"power", "50hz", "powerline"}:
        t = np.arange(length, dtype=np.float32) / float(sr)
        noise = np.sin(2 * np.pi * 50.0 * t)
    elif kind == "mixed":
        parts = [
            generate_noise("white", length, sr, rng),
            generate_noise("pink", length, sr, rng),
            generate_noise("power", length, sr, rng),
        ]
        noise = sum(part / max(rms(part), EPS) for part in parts)
    else:
        raise ValueError(f"Unknown noise kind: {kind}")

    return normalize_audio(np.asarray(noise, dtype=np.float32))


def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms <= EPS:
        return clean.copy()
    scale = clean_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
    return (clean + noise * scale).astype(np.float32)


def speaker_id_from_path(path: str | Path) -> int:
    stem = Path(path).stem.lower()
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer speaker id from {path}")
    return int(digits)


def resolve_speaker_files(reference_dir: str | Path, speaker_ids: Iterable[int]) -> dict[int, Path]:
    reference_dir = Path(reference_dir)
    result: dict[int, Path] = {}
    for speaker_id in speaker_ids:
        matches = sorted(reference_dir.glob(f"speaker{speaker_id}.*"))
        if not matches:
            matches = sorted(reference_dir.glob(f"speaker_{speaker_id}.*"))
        if not matches:
            raise FileNotFoundError(f"Missing audio file for speaker{speaker_id} in {reference_dir}")
        result[speaker_id] = matches[0]
    return result

