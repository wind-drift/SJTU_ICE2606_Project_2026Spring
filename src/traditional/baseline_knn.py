import os
import numpy as np
import pandas as pd
import soundfile as sf

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.fftpack import dct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


METADATA_PATH = "data/processed/metadata.csv"
OUTPUT_DIR = "results/traditional/baseline_knn"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

NUM_CLASSES = 10
TARGET_LABELS = list(range(NUM_CLASSES))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_wav(path):
    """
    Load wav file and normalize to [-1, 1].
    """
    y, sr = sf.read(path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float32)

    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs

    return y, sr


def pre_emphasis(y, coeff=0.97):
    """
    Pre-emphasis:
        y[n] = x[n] - coeff * x[n-1]
    """
    if len(y) == 0:
        return y

    return np.append(y[0], y[1:] - coeff * y[:-1])


def framing(y, sr, frame_size=0.025, frame_stride=0.010):
    """
    Split signal into overlapping short-time frames.
    Default:
        25 ms frame length
        10 ms frame stride
    """
    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))

    signal_length = len(y)

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

    pad_length = (num_frames - 1) * frame_step + frame_length
    pad_signal = np.append(y, np.zeros(pad_length - signal_length))

    indices = (
        np.tile(np.arange(frame_length), (num_frames, 1))
        + np.tile(np.arange(num_frames) * frame_step, (frame_length, 1)).T
    )

    frames = pad_signal[indices]
    return frames, frame_length, frame_step


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_filter_bank(sr, nfft=512, nfilt=26):
    """
    Construct mel filter bank.
    """
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, nfft // 2 + 1))

    for m in range(1, nfilt + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        for k in range(left, center):
            if center != left:
                fbank[m - 1, k] = (k - left) / (center - left)

        for k in range(center, right):
            if right != center:
                fbank[m - 1, k] = (right - k) / (right - center)

    return fbank


def compute_mfcc(
    y,
    sr,
    num_ceps=13,
    nfilt=26,
    nfft=512,
    pre_emphasis_coeff=0.97,
    frame_size=0.025,
    frame_stride=0.010,
):
    """
    Manual MFCC extraction:
    pre-emphasis -> framing -> Hamming window -> FFT
    -> mel filter bank -> log energy -> DCT -> first 13 coefficients.
    """
    emphasized = pre_emphasis(y, pre_emphasis_coeff)

    frames, frame_length, _ = framing(
        emphasized,
        sr,
        frame_size=frame_size,
        frame_stride=frame_stride,
    )

    frames *= np.hamming(frame_length)

    magnitude = np.abs(np.fft.rfft(frames, nfft))
    power = (1.0 / nfft) * (magnitude ** 2)

    fbank = mel_filter_bank(sr, nfft=nfft, nfilt=nfilt)
    filter_banks = np.dot(power, fbank.T)

    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks,
    )

    log_filter_banks = np.log(filter_banks)

    mfcc = dct(
        log_filter_banks,
        type=2,
        axis=1,
        norm="ortho",
    )[:, :num_ceps]

    return mfcc


def extract_statistical_feature(path):
    """
    Convert one wav file into a fixed-length feature vector.

    Feature:
        mean of 13 MFCC coefficients
        std of 13 MFCC coefficients

    Total dimension:
        13 + 13 = 26
    """
    y, sr = load_wav(path)
    mfcc = compute_mfcc(y, sr)

    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)

    feature = np.concatenate([mfcc_mean, mfcc_std])
    return feature


def load_dataset(metadata_path):
    """
    Load metadata.csv and extract features for all samples.
    """
    metadata = pd.read_csv(metadata_path)

    features = []
    labels = []
    speakers = []
    splits = []
    paths = []

    for _, row in metadata.iterrows():
        path = row["path"]
        label = int(row["label"])
        speaker = int(row["speaker"])
        split = row["split"]

        feature = extract_statistical_feature(path)

        features.append(feature)
        labels.append(label)
        speakers.append(speaker)
        splits.append(split)
        paths.append(path)

    X = np.vstack(features)
    y = np.array(labels)

    info = pd.DataFrame({
        "path": paths,
        "speaker": speakers,
        "label": labels,
        "split": splits,
    })

    return X, y, info


def plot_confusion_matrix(cm, output_path):
    """
    Plot confusion matrix with a clean blue-white style for slides.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.4))

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=max(1, cm.max()),
    )

    ax.set_title("Clean Confusion Matrix (MFCC-stat + KNN)", fontsize=11, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)

    ax.set_xticks(TARGET_LABELS)
    ax.set_yticks(TARGET_LABELS)
    ax.tick_params(axis="both", labelsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(X, y, info):
    """
    Train KNN on train split and evaluate on val split.
    """
    train_mask = info["split"].values == "train"
    val_mask = info["split"].values == "val"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    val_info = info[val_mask].copy().reset_index(drop=True)

    print("Dataset summary:")
    print(f"  Train samples: {len(y_train)}")
    print(f"  Val samples:   {len(y_val)}")
    print(f"  Feature dim:   {X.shape[1]}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=1, metric="euclidean")),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred, labels=TARGET_LABELS)
    report = classification_report(
        y_val,
        y_pred,
        labels=TARGET_LABELS,
        digits=4,
        zero_division=0,
    )

    print("\nClean validation accuracy:")
    print(f"  {acc:.4f}")

    print("\nClassification report:")
    print(report)

    val_info["prediction"] = y_pred
    val_info["correct"] = val_info["label"] == val_info["prediction"]

    return model, acc, cm, report, val_info


def save_results(acc, cm, report, val_info):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(TABLE_DIR)
    ensure_dir(METRIC_DIR)

    cm_png_path = os.path.join(FIGURE_DIR, "clean_confusion_matrix.png")
    cm_csv_path = os.path.join(TABLE_DIR, "clean_confusion_matrix.csv")
    pred_csv_path = os.path.join(TABLE_DIR, "clean_predictions.csv")
    metrics_path = os.path.join(METRIC_DIR, "clean_metrics.txt")

    plot_confusion_matrix(cm, cm_png_path)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in TARGET_LABELS],
        columns=[f"pred_{i}" for i in TARGET_LABELS],
    )
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig", index=True)

    val_info.to_csv(pred_csv_path, encoding="utf-8-sig", index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Clean validation accuracy\n")
        f.write(f"{acc:.6f}\n\n")
        f.write("Classification report\n")
        f.write(report)
        f.write("\n")

    print("\nSaved results:")
    print(f"  {cm_png_path}")
    print(f"  {cm_csv_path}")
    print(f"  {pred_csv_path}")
    print(f"  {metrics_path}")


def main():
    print("Loading dataset and extracting MFCC statistical features...")
    X, y, info = load_dataset(METADATA_PATH)

    model, acc, cm, report, val_info = train_and_evaluate(X, y, info)

    save_results(acc, cm, report, val_info)

    print("\nBaseline KNN finished.")


if __name__ == "__main__":
    main()
    