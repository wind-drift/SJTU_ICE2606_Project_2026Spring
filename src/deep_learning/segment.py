"""Energy-based segmentation for the ten digits in each speaker recording."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DIGIT_LABELS, FeatureConfig, TRAIN_SPEAKERS, VAL_SPEAKERS
from .features import load_audio, resolve_speaker_files, speaker_id_from_path


@dataclass
class DigitSegment:
    source_path: str
    speaker_id: int
    label: int
    start_sample: int
    end_sample: int
    sr: int

    @property
    def start_sec(self) -> float:
        return self.start_sample / float(self.sr)

    @property
    def end_sec(self) -> float:
        return self.end_sample / float(self.sr)

    @property
    def duration_sec(self) -> float:
        return (self.end_sample - self.start_sample) / float(self.sr)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "source_path": self.source_path,
            "speaker_id": self.speaker_id,
            "label": self.label,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "sr": self.sr,
            "start_sec": round(self.start_sec, 4),
            "end_sec": round(self.end_sec, 4),
            "duration_sec": round(self.duration_sec, 4),
        }


@dataclass
class DigitSample:
    metadata: DigitSegment
    waveform: np.ndarray

    @property
    def label(self) -> int:
        return self.metadata.label


def _frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(audio) < frame_length:
        padded = np.pad(audio, (0, frame_length - len(audio)))
        return np.array([np.sqrt(np.mean(padded * padded))], dtype=np.float32)
    starts = range(0, len(audio) - frame_length + 1, hop_length)
    values = [np.sqrt(np.mean(audio[start : start + frame_length] ** 2)) for start in starts]
    return np.asarray(values, dtype=np.float32)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) <= 1:
        return values
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values, kernel, mode="same")


def _mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


def _merge_close(intervals: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    if not intervals:
        return intervals
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_frames:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _remove_short(intervals: list[tuple[int, int]], min_frames: int) -> list[tuple[int, int]]:
    return [(start, end) for start, end in intervals if end - start >= min_frames]


def _coerce_count(
    intervals: list[tuple[int, int]],
    expected_count: int,
    total_frames: int,
) -> list[tuple[int, int]]:
    if len(intervals) == expected_count:
        return intervals

    if len(intervals) > expected_count:
        intervals = intervals[:]
        while len(intervals) > expected_count:
            gaps = [intervals[idx + 1][0] - intervals[idx][1] for idx in range(len(intervals) - 1)]
            merge_idx = int(np.argmin(gaps))
            merged = (intervals[merge_idx][0], intervals[merge_idx + 1][1])
            intervals[merge_idx : merge_idx + 2] = [merged]
        return intervals

    if intervals:
        start = intervals[0][0]
        end = intervals[-1][1]
    else:
        start = 0
        end = total_frames
    end = max(end, start + expected_count)
    edges = np.linspace(start, end, expected_count + 1).round().astype(int)
    return [(int(edges[idx]), int(edges[idx + 1])) for idx in range(expected_count)]


def find_digit_intervals(
    audio: np.ndarray,
    sr: int,
    expected_count: int = 10,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_digit_ms: float = 120.0,
    merge_gap_ms: float = 180.0,
    margin_ms: float = 100.0,
) -> list[tuple[int, int]]:
    frame_length = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sr * hop_ms / 1000.0)))
    rms = _frame_rms(audio, frame_length, hop_length)
    if np.max(rms) <= 1e-8:
        edges = np.linspace(0, len(audio), expected_count + 1).round().astype(int)
        return [(int(edges[idx]), int(edges[idx + 1])) for idx in range(expected_count)]

    energy = rms / (np.max(rms) + 1e-8)
    energy = _smooth(energy, width=max(1, int(round(60.0 / hop_ms))))
    low = float(np.percentile(energy, 20))
    high = float(np.percentile(energy, 95))
    threshold = max(0.03, low + 0.25 * (high - low))

    mask = energy > threshold
    intervals = _mask_to_intervals(mask)
    intervals = _merge_close(intervals, max_gap_frames=max(1, int(round(merge_gap_ms / hop_ms))))
    intervals = _remove_short(intervals, min_frames=max(1, int(round(min_digit_ms / hop_ms))))
    intervals = _coerce_count(intervals, expected_count, len(energy))

    margin = int(round(sr * margin_ms / 1000.0))
    sample_intervals: list[tuple[int, int]] = []
    for start_frame, end_frame in intervals:
        start = max(0, start_frame * hop_length - margin)
        end = min(len(audio), max(start + 1, (end_frame - 1) * hop_length + frame_length + margin))
        sample_intervals.append((start, end))
    return sample_intervals


def split_speaker_file(
    path: str | Path,
    config: FeatureConfig | None = None,
    labels: Sequence[int] = DIGIT_LABELS,
) -> list[DigitSample]:
    config = config or FeatureConfig()
    path = Path(path)
    audio, sr = load_audio(path, target_sr=config.sr)
    speaker_id = speaker_id_from_path(path)
    intervals = find_digit_intervals(audio, sr, expected_count=len(labels))
    samples: list[DigitSample] = []
    for label, (start, end) in zip(labels, intervals):
        metadata = DigitSegment(
            source_path=path.as_posix(),
            speaker_id=speaker_id,
            label=int(label),
            start_sample=int(start),
            end_sample=int(end),
            sr=sr,
        )
        samples.append(DigitSample(metadata=metadata, waveform=audio[start:end].copy()))
    return samples


def load_digit_samples(
    reference_dir: str | Path = "reference",
    speaker_ids: Sequence[int] = TRAIN_SPEAKERS + VAL_SPEAKERS,
    config: FeatureConfig | None = None,
) -> list[DigitSample]:
    config = config or FeatureConfig()
    files = resolve_speaker_files(reference_dir, speaker_ids)
    samples: list[DigitSample] = []
    for speaker_id in speaker_ids:
        samples.extend(split_speaker_file(files[speaker_id], config=config))
    return samples


def write_manifest(samples: Sequence[DigitSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([sample.metadata.to_dict() for sample in samples], indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect energy-based digit segmentation.")
    parser.add_argument("--reference-dir", default="reference")
    parser.add_argument("--output", default="outputs/deep_learning/segments_manifest.json")
    parser.add_argument(
        "--speaker-ids",
        type=int,
        nargs="+",
        default=list(TRAIN_SPEAKERS + VAL_SPEAKERS),
    )
    args = parser.parse_args()

    samples = load_digit_samples(args.reference_dir, tuple(args.speaker_ids))
    write_manifest(samples, args.output)
    by_speaker: dict[int, int] = {}
    for sample in samples:
        by_speaker[sample.metadata.speaker_id] = by_speaker.get(sample.metadata.speaker_id, 0) + 1
    print(f"Wrote {len(samples)} segments to {args.output}")
    print("Segments per speaker:", by_speaker)


if __name__ == "__main__":
    main()

