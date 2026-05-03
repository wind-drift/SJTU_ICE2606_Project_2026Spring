"""Metrics and output helpers for digit recognition experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if len(true) == 0:
        return 0.0
    return float(np.mean(true == pred))


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 10) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_confusion_csv(matrix: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *range(matrix.shape[1])])
        for idx, row in enumerate(matrix):
            writer.writerow([idx, *row.tolist()])


def plot_confusion_matrix(matrix: np.ndarray, path: str | Path, title: str = "Confusion Matrix") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_accuracy_curve(rows: list[dict[str, float | str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        by_kind.setdefault(str(row["noise_kind"]), []).append((float(row["snr_db"]), float(row["accuracy"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, points in sorted(by_kind.items()):
        points = sorted(points)
        snrs = [point[0] for point in points]
        accs = [point[1] * 100.0 for point in points]
        ax.plot(snrs, accs, marker="o", label=kind)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Noise robustness")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

