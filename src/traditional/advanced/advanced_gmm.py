"""
tradition method, advanced
using mfcc+delta+delta^2 and gmm
Jing Xu
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.fftpack import dct
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

METADATA_PATH = "data/processed/metadata.csv"
NOISE_PATH = "data/noise/mixed_noise.wav"
OUTPUT_DIR = "results/traditional/advanced_gmm"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

NUM_CLASSES = 10
TARGET_LABELS = list(range(NUM_CLASSES))
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
NOISE_REPEATS = 20

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_wav(path):
    y, sr = sf.read(path)
    if y.ndim > 1: y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    max_abs = np.max(np.abs(y))
    if max_abs > 0: y = y / max_abs
    return y, sr

def pre_emphasis(y, coeff=0.97):
    return np.append(y[0], y[1:] - coeff * y[:-1]) if len(y) > 0 else y

def framing(y, sr, frame_size=0.025, frame_stride=0.010):
    frame_length, frame_step = int(round(frame_size * sr)), int(round(frame_stride * sr))
    signal_length = len(y)
    num_frames = 1 if signal_length <= frame_length else int(np.ceil((signal_length - frame_length) / frame_step)) + 1
    pad_length = (num_frames - 1) * frame_step + frame_length
    pad_signal = np.append(y, np.zeros(pad_length - signal_length))
    indices = np.tile(np.arange(frame_length), (num_frames, 1)) + np.tile(np.arange(num_frames) * frame_step, (frame_length, 1)).T
    return pad_signal[indices], frame_length, frame_step

def mel_filter_bank(sr, nfft=512, nfilt=26):
    low_mel, high_mel = 0, 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((nfilt, nfft // 2 + 1))
    for m in range(1, nfilt + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(left, center): fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right): fbank[m - 1, k] = (right - k) / (right - center)
    return fbank

def compute_mfcc(y, sr, num_ceps=13, nfilt=26, nfft=512):
    emphasized = pre_emphasis(y)
    frames, frame_length, _ = framing(emphasized, sr)
    frames *= np.hamming(frame_length)
    magnitude = np.abs(np.fft.rfft(frames, nfft))
    power = (1.0 / nfft) * (magnitude ** 2)
    fbank = mel_filter_bank(sr, nfft=nfft, nfilt=nfilt)
    filter_banks = np.dot(power, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_filter_banks = np.log(filter_banks)
    mfcc = dct(log_filter_banks, type=2, axis=1, norm="ortho")[:, :num_ceps]
    return mfcc

def compute_delta(feat, N=2):
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    delta = np.zeros_like(feat)
    denominator = sum([2 * (i**2) for i in range(1, N+1)])
    for t in range(len(feat)):
        delta[t] = sum([i * (padded[t+N+i] - padded[t+N-i]) for i in range(1, N+1)]) / denominator
    return delta

def extract_advanced_features(path):
    y, sr = load_wav(path)

    mfcc = compute_mfcc(y, sr)

    delta1 = compute_delta(mfcc, N=2)
    delta2 = compute_delta(delta1, N=2)

    feat_39 = np.concatenate([mfcc, delta1, delta2], axis=1)

    feat_cmvn = (feat_39 - np.mean(feat_39, axis=0)) / (np.std(feat_39, axis=0) + 1e-8)
    
    return feat_cmvn

def load_dataset(metadata_path):
    metadata = pd.read_csv(metadata_path)
    features, labels, splits, paths = [], [], [], []
    
    for _, row in metadata.iterrows():
        path = row["path"]
        feat = extract_advanced_features(path)
        features.append(feat)
        labels.append(int(row["label"]))
        splits.append(row["split"])
        paths.append(path)
        
    info = pd.DataFrame({"path": paths, "label": labels, "split": splits})
    return features, np.array(labels), info

class GMMClassifier:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.models = {}

    def fit(self, X_train, y_train):
        print(f"Training GMMs...")
        for c in TARGET_LABELS:
            X_c = np.vstack([X_train[i] for i in range(len(X_train)) if y_train[i] == c])
            
            gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag', random_state=42)
            gmm.fit(X_c)
            self.models[c] = gmm
            
    def predict(self, X_val):
        predictions = []
        for x in X_val:
            scores = [self.models[c].score(x) for c in TARGET_LABELS]
            predictions.append(np.argmax(scores))
        return np.array(predictions)

def mix_noise(y_clean, y_noise, snr):
    if len(y_noise) < len(y_clean):
        y_noise = np.tile(y_noise, int(np.ceil(len(y_clean)/len(y_noise))))

    start = np.random.randint(0, len(y_noise) - len(y_clean) + 1)
    noise_segment = y_noise[start:start + len(y_clean)]

    p_clean = np.mean(y_clean ** 2) + 1e-8
    p_noise = np.mean(noise_segment ** 2) + 1e-8
    
    alpha = np.sqrt(p_clean / (p_noise * (10 ** (snr / 10.0))))
    y_noisy = y_clean + alpha * noise_segment
    
    max_val = np.max(np.abs(y_noisy))
    if max_val > 1.0: y_noisy /= max_val
    return y_noisy

def evaluate_noise_robustness(model, val_info, noise_wav_path):
    print("\nStarting repeated noise robustness test...")
    y_noise, _ = load_wav(noise_wav_path)
    
    results = []
    
    for snr in SNR_LEVELS:
        snr_accs = []
        for rep in range(NOISE_REPEATS):
            noisy_features = []
            y_true = []
            for _, row in val_info.iterrows():

                y_clean, sr = load_wav(row["path"])
                y_noisy = mix_noise(y_clean, y_noise, snr)

                mfcc = compute_mfcc(y_noisy, sr)
                d1 = compute_delta(mfcc, N=2)
                d2 = compute_delta(d1, N=2)
                feat = np.concatenate([mfcc, d1, d2], axis=1)
                feat = (feat - np.mean(feat, axis=0)) / (np.std(feat, axis=0) + 1e-8)
                
                noisy_features.append(feat)
                y_true.append(row["label"])
                
            y_pred = model.predict(noisy_features)
            acc = accuracy_score(y_true, y_pred)
            snr_accs.append(acc)
            
        mean_acc = np.mean(snr_accs)
        std_acc = np.std(snr_accs)
        print(f"SNR {snr:3d} dB -> Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
        results.append((snr, mean_acc, std_acc))
        
    return results

def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Clean Confusion Matrix (MFCC-39 + CMVN + GMM)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(TARGET_LABELS)
    plt.yticks(TARGET_LABELS)
    plt.colorbar(label="Count")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > np.max(cm)/2 else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_snr_curve(results, output_path):
    snrs = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(snrs, means, yerr=stds, fmt='-o', capsize=5, label='Advanced GMM')
    plt.title("Noise Robustness: Accuracy vs SNR (20 Repeats)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(TABLE_DIR)
    ensure_dir(METRIC_DIR)

    print("Loading...")
    X_all, y_all, info = load_dataset(METADATA_PATH)
    
    train_mask = info["split"] == "train"
    val_mask = info["split"] == "val"
    
    X_train = [X_all[i] for i in range(len(X_all)) if train_mask[i]]
    y_train = y_all[train_mask]
    X_val = [X_all[i] for i in range(len(X_all)) if val_mask[i]]
    y_val = y_all[val_mask]
    val_info = info[val_mask].copy()
    
    model = GMMClassifier(n_components=4)
    model.fit(X_train, y_train)
    
    y_pred_clean = model.predict(X_val)
    clean_acc = accuracy_score(y_val, y_pred_clean)
    cm = confusion_matrix(y_val, y_pred_clean)
    report = classification_report(y_val, y_pred_clean, digits=4)
    
    print(f"\nClean Accuracy: {clean_acc:.4f}")
    
    plot_confusion_matrix(cm, os.path.join(FIGURE_DIR, "clean_confusion_matrix.png"))
    with open(os.path.join(METRIC_DIR, "clean_metrics.txt"), "w") as f:
        f.write(f"Clean validation accuracy\n{clean_acc:.6f}\n\nClassification report\n{report}\n")
        
    if os.path.exists(NOISE_PATH):
        noise_results = evaluate_noise_robustness(model, val_info, NOISE_PATH)
        plot_snr_curve(noise_results, os.path.join(FIGURE_DIR, "accuracy_snr_curve.png"))
        
        df_noise = pd.DataFrame(noise_results, columns=["SNR", "Mean_Acc", "Std_Acc"])
        df_noise.to_csv(os.path.join(TABLE_DIR, "snr_accuracy_summary.csv"), index=False)
    else:
        print(f"\nNoise file not found at {NOISE_PATH}.")

    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
