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

NOISE_PATH_CANDIDATES = [
    "data/noise/mixed_noise.wav",
    "data/raw/mixed_noise.wav",
    "mixed_noise.wav",
    "docs/traditional/mixed_noise.wav",
]

OUTPUT_DIR = "results/traditional/noise_robustness_v2"

SNR_LIST = [-10, -5, 0, 5, 10, 15, 20]
N_REPEATS = 20
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
    Get a fixed noise segment with the same length as the clean speech.
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


def load_validation_samples(val_metadata):
    samples = []

    for idx, row in val_metadata.iterrows():
        clean, sr = load_wav(row["path"], normalize=True)

        samples.append({
            "index": idx,
            "path": row["path"],
            "speaker": int(row["speaker"]),
            "label": int(row["label"]),
            "clean": clean,
            "sr": sr,
            "length": len(clean),
        })

    return samples


def evaluate_clean_reference(model, val_samples):
    features = []
    labels = []

    for sample in val_samples:
        feature = extract_statistical_feature_from_signal(
            sample["clean"],
            sample["sr"],
        )
        features.append(feature)
        labels.append(sample["label"])

    X_val = np.vstack(features)
    y_true = np.array(labels)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_true, y_pred)

    return acc, y_true, y_pred


def make_fixed_noise_segments(val_samples, noise, rng):
    """
    For one repeat, each validation sample gets one fixed noise segment.
    The same segment will be reused across all SNR values.
    """
    noise_segments = {}

    for sample in val_samples:
        noise_segments[sample["index"]] = get_noise_segment(
            noise,
            sample["length"],
            rng,
        )

    return noise_segments


def evaluate_one_repeat(model, val_samples, noise_segments, snr_db, repeat_idx):
    features = []
    labels = []
    rows = []
    actual_snr_values = []

    for sample in val_samples:
        clean = sample["clean"]
        sr = sample["sr"]
        noise_segment = noise_segments[sample["index"]]

        noisy, actual_snr = add_noise_at_snr(clean, noise_segment, snr_db)

        feature = extract_statistical_feature_from_signal(noisy, sr)

        features.append(feature)
        labels.append(sample["label"])
        actual_snr_values.append(actual_snr)

        rows.append({
            "repeat": repeat_idx,
            "snr_db": snr_db,
            "path": sample["path"],
            "speaker": sample["speaker"],
            "label": sample["label"],
            "actual_snr_db": actual_snr,
        })

    X_val_noisy = np.vstack(features)
    y_true = np.array(labels)

    y_pred = model.predict(X_val_noisy)
    acc = accuracy_score(y_true, y_pred)

    result_df = pd.DataFrame(rows)
    result_df["prediction"] = y_pred
    result_df["correct"] = result_df["label"] == result_df["prediction"]

    mean_actual_snr = float(np.mean(actual_snr_values))

    return acc, mean_actual_snr, result_df


def plot_accuracy_snr_curve(summary_df, clean_acc, output_path):
    plt.figure(figsize=(7, 4))

    plt.errorbar(
        summary_df["snr_db"],
        summary_df["accuracy_mean"],
        yerr=summary_df["accuracy_std"],
        marker="o",
        capsize=4,
        label="Noisy validation",
    )

    plt.axhline(
        clean_acc,
        linestyle="--",
        label=f"Clean reference = {clean_acc:.2f}",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Noise Robustness: Accuracy vs. SNR")
    plt.xticks(SNR_LIST)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_results(summary_rows, all_repeat_rows, clean_acc):
    ensure_dir(OUTPUT_DIR)

    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.concat(all_repeat_rows, ignore_index=True)

    summary_csv_path = os.path.join(OUTPUT_DIR, "snr_accuracy_summary.csv")
    pred_csv_path = os.path.join(OUTPUT_DIR, "noise_predictions_repeated.csv")
    curve_path = os.path.join(OUTPUT_DIR, "accuracy_snr_curve_mean_std.png")
    metrics_path = os.path.join(OUTPUT_DIR, "noise_test_repeated_metrics.txt")

    summary_df.to_csv(summary_csv_path, encoding="utf-8-sig", index=False)
    prediction_df.to_csv(pred_csv_path, encoding="utf-8-sig", index=False)

    plot_accuracy_snr_curve(summary_df, clean_acc, curve_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Repeated noise robustness test\n\n")
        f.write(f"N_REPEATS = {N_REPEATS}\n")
        f.write(f"Clean reference accuracy = {clean_acc:.6f}\n\n")
        f.write("SNR(dB), Accuracy mean, Accuracy std, Mean actual SNR(dB)\n")

        for row in summary_rows:
            f.write(
                f"{row['snr_db']}, "
                f"{row['accuracy_mean']:.6f}, "
                f"{row['accuracy_std']:.6f}, "
                f"{row['mean_actual_snr_db']:.6f}\n"
            )

    print("\nSaved results:")
    print(f"  {summary_csv_path}")
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

    print("\nLoading validation samples...")
    val_samples = load_validation_samples(val_metadata)

    clean_acc, clean_true, clean_pred = evaluate_clean_reference(model, val_samples)
    print(f"Clean reference accuracy in this script: {clean_acc:.4f}")

    target_sr = val_samples[0]["sr"]
    noise, noise_sr = load_noise(target_sr)

    if noise_sr != target_sr:
        raise RuntimeError("Noise sampling rate does not match target sampling rate.")

    summary_rows = []
    all_repeat_rows = []

    print("\nRunning repeated noise robustness test...")
    print(f"Repeats per SNR: {N_REPEATS}")

    for repeat_idx in range(N_REPEATS):
        rng = np.random.default_rng(RANDOM_SEED + repeat_idx)
        noise_segments = make_fixed_noise_segments(val_samples, noise, rng)

        for snr_db in SNR_LIST:
            acc, mean_actual_snr, result_df = evaluate_one_repeat(
                model,
                val_samples,
                noise_segments,
                snr_db,
                repeat_idx,
            )

            all_repeat_rows.append(result_df)

        print(f"  Repeat {repeat_idx + 1:>2}/{N_REPEATS} finished.")

    all_prediction_df = pd.concat(all_repeat_rows, ignore_index=True)

    for snr_db in SNR_LIST:
        snr_df = all_prediction_df[all_prediction_df["snr_db"] == snr_db]

        # Accuracy for each repeat
        repeat_acc = (
            snr_df
            .groupby("repeat")["correct"]
            .mean()
            .values
        )

        actual_snr_mean = float(snr_df["actual_snr_db"].mean())

        acc_mean = float(np.mean(repeat_acc))
        acc_std = float(np.std(repeat_acc, ddof=1)) if len(repeat_acc) > 1 else 0.0

        summary_rows.append({
            "snr_db": snr_db,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "mean_actual_snr_db": actual_snr_mean,
        })

        print(
            f"  SNR = {snr_db:>3} dB | "
            f"Accuracy = {acc_mean:.4f} ± {acc_std:.4f} | "
            f"Mean actual SNR = {actual_snr_mean:.3f} dB"
        )

    save_results(summary_rows, [all_prediction_df], clean_acc)

    print("\nRepeated noise robustness test finished.")


if __name__ == "__main__":
    main()
    