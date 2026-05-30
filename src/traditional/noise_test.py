import os

# Reduce possible OpenMP conflicts in small experiments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import soundfile as sf

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.fftpack import dct
from scipy import signal

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


METADATA_PATH = "data/processed/metadata.csv"

# The script will search these paths in order.
NOISE_PATH_CANDIDATES = [
    "data/raw/mixed_noise.wav",
    "mixed_noise.wav",
    "docs/traditional/mixed_noise.wav",
    "data/noise/mixed_noise.wav"
]

OUTPUT_DIR = "results/traditional/noise_robustness"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

SNR_LIST = [-10, -5, 0, 5, 10, 15, 20]
TARGET_LABELS = list(range(10))
RANDOM_SEED = 2026


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_noise_path():
    for path in NOISE_PATH_CANDIDATES:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Cannot find mixed_noise.wav. Please put it in one of these paths:\n"
        + "\n".join(NOISE_PATH_CANDIDATES)
    )


def load_wav(path, normalize=True):
    y, sr = sf.read(path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float32)

    if normalize:
        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            y = y / max_abs

    return y, sr


def load_noise(target_sr):
    noise_path = find_noise_path()
    noise, noise_sr = load_wav(noise_path, normalize=True)

    if noise_sr != target_sr:
        gcd = np.gcd(noise_sr, target_sr)
        up = target_sr // gcd
        down = noise_sr // gcd
        noise = signal.resample_poly(noise, up, down)
        noise_sr = target_sr

    print(f"Loaded noise: {noise_path}")
    print(f"Noise sampling rate: {noise_sr} Hz")
    print(f"Noise duration: {len(noise) / noise_sr:.3f} s")

    return noise, noise_sr


def get_noise_segment(noise, target_length, rng):
    """
    Get a noise segment with the same length as the clean speech.
    If the noise is shorter than target_length, repeat it.
    """
    if len(noise) >= target_length:
        max_start = len(noise) - target_length
        start = rng.integers(0, max_start + 1)
        return noise[start:start + target_length]

    repeat_times = int(np.ceil(target_length / len(noise)))
    tiled_noise = np.tile(noise, repeat_times)
    return tiled_noise[:target_length]


def add_noise_at_snr(clean, noise_segment, snr_db):
    """
    Add noise to clean speech at a target SNR.

    SNR = 10 * log10(P_signal / P_noise)

    To reach target SNR:
        alpha = sqrt(P_signal / (P_noise * 10^(SNR/10)))
        noisy = clean + alpha * noise
    """
    clean = clean.astype(np.float32)
    noise_segment = noise_segment.astype(np.float32)

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise_segment ** 2)

    if signal_power <= 0:
        return clean.copy(), 0.0

    if noise_power <= 0:
        raise ValueError("Noise segment has zero power.")

    target_linear = 10 ** (snr_db / 10)
    alpha = np.sqrt(signal_power / (noise_power * target_linear))

    scaled_noise = alpha * noise_segment
    noisy = clean + scaled_noise

    actual_noise_power = np.mean(scaled_noise ** 2)
    actual_snr = 10 * np.log10(signal_power / actual_noise_power)

    return noisy, actual_snr


def pre_emphasis(y, coeff=0.97):
    if len(y) == 0:
        return y

    return np.append(y[0], y[1:] - coeff * y[:-1])


def framing(y, sr, frame_size=0.025, frame_stride=0.010):
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


def extract_statistical_feature_from_signal(y, sr):
    mfcc = compute_mfcc(y, sr)

    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)

    return np.concatenate([mfcc_mean, mfcc_std])


def extract_statistical_feature_from_path(path):
    y, sr = load_wav(path, normalize=True)
    return extract_statistical_feature_from_signal(y, sr)


def load_clean_train_data(metadata):
    train_metadata = metadata[metadata["split"] == "train"].copy()

    features = []
    labels = []

    for _, row in train_metadata.iterrows():
        feature = extract_statistical_feature_from_path(row["path"])
        features.append(feature)
        labels.append(int(row["label"]))

    X_train = np.vstack(features)
    y_train = np.array(labels)

    return X_train, y_train


def train_clean_model(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=1, metric="euclidean")),
    ])

    model.fit(X_train, y_train)
    return model


def evaluate_at_snr(model, val_metadata, noise, snr_db, rng):
    features = []
    labels = []
    rows = []

    actual_snr_values = []

    for _, row in val_metadata.iterrows():
        clean, sr = load_wav(row["path"], normalize=True)

        noise_segment = get_noise_segment(noise, len(clean), rng)
        noisy, actual_snr = add_noise_at_snr(clean, noise_segment, snr_db)

        feature = extract_statistical_feature_from_signal(noisy, sr)

        features.append(feature)
        labels.append(int(row["label"]))

        rows.append({
            "snr_db": snr_db,
            "path": row["path"],
            "speaker": int(row["speaker"]),
            "label": int(row["label"]),
            "actual_snr_db": actual_snr,
        })

        actual_snr_values.append(actual_snr)

    X_val_noisy = np.vstack(features)
    y_true = np.array(labels)

    y_pred = model.predict(X_val_noisy)

    acc = accuracy_score(y_true, y_pred)

    result_df = pd.DataFrame(rows)
    result_df["prediction"] = y_pred
    result_df["correct"] = result_df["label"] == result_df["prediction"]

    mean_actual_snr = float(np.mean(actual_snr_values))

    return acc, mean_actual_snr, result_df


def plot_accuracy_snr_curve(snr_accuracy_df, output_path):
    plt.figure(figsize=(7, 4))

    plt.plot(
        snr_accuracy_df["snr_db"],
        snr_accuracy_df["accuracy"],
        marker="o",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Noise Robustness: Accuracy vs. SNR")
    plt.xticks(SNR_LIST)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_results(snr_rows, all_predictions):
    ensure_dir(OUTPUT_DIR)

    snr_accuracy_df = pd.DataFrame(snr_rows)
    prediction_df = pd.concat(all_predictions, ignore_index=True)

    snr_csv_path = os.path.join(TABLE_DIR, "snr_accuracy.csv")
    pred_csv_path = os.path.join(TABLE_DIR, "noise_predictions.csv")
    curve_path = os.path.join(FIGURE_DIR, "accuracy_snr_curve.png")
    metrics_path = os.path.join(METRIC_DIR, "noise_test_metrics.txt")

    snr_accuracy_df.to_csv(snr_csv_path, encoding="utf-8-sig", index=False)
    prediction_df.to_csv(pred_csv_path, encoding="utf-8-sig", index=False)

    plot_accuracy_snr_curve(snr_accuracy_df, curve_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Noise robustness test\n\n")
        f.write("SNR(dB), Accuracy, Mean actual SNR(dB)\n")
        for row in snr_rows:
            f.write(
                f"{row['snr_db']}, "
                f"{row['accuracy']:.6f}, "
                f"{row['mean_actual_snr_db']:.6f}\n"
            )

    print("\nSaved results:")
    print(f"  {snr_csv_path}")
    print(f"  {pred_csv_path}")
    print(f"  {curve_path}")
    print(f"  {metrics_path}")


def main():
    print("Loading metadata...")
    metadata = pd.read_csv(METADATA_PATH)

    train_metadata = metadata[metadata["split"] == "train"]
    val_metadata = metadata[metadata["split"] == "val"].copy().reset_index(drop=True)

    print("Dataset summary:")
    print(f"  Train samples: {len(train_metadata)}")
    print(f"  Val samples:   {len(val_metadata)}")

    print("\nExtracting clean training features...")
    X_train, y_train = load_clean_train_data(metadata)

    print("Training clean MFCC-stat + KNN model...")
    model = train_clean_model(X_train, y_train)

    # Load one validation sample to get target sr
    sample_y, target_sr = load_wav(val_metadata.iloc[0]["path"], normalize=True)
    noise, noise_sr = load_noise(target_sr)

    if noise_sr != target_sr:
        raise RuntimeError("Noise sampling rate does not match target sampling rate.")

    snr_rows = []
    all_predictions = []

    print("\nRunning noise robustness test...")

    for snr_db in SNR_LIST:
        rng = np.random.default_rng(RANDOM_SEED + snr_db + 100)

        acc, mean_actual_snr, result_df = evaluate_at_snr(
            model,
            val_metadata,
            noise,
            snr_db,
            rng,
        )

        snr_rows.append({
            "snr_db": snr_db,
            "accuracy": acc,
            "mean_actual_snr_db": mean_actual_snr,
        })

        all_predictions.append(result_df)

        print(
            f"  SNR = {snr_db:>3} dB | "
            f"Accuracy = {acc:.4f} | "
            f"Mean actual SNR = {mean_actual_snr:.3f} dB"
        )

    save_results(snr_rows, all_predictions)

    print("\nNoise robustness test finished.")


if __name__ == "__main__":
    main()
