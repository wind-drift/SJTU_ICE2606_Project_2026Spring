import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import dct


SAMPLE_PATH = "data/processed/train/speaker1_digit0.wav"
OUTPUT_DIR = "results/traditional/signal_analysis/figures"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_wav(path):
    """
    Load wav file and normalize it to [-1, 1].
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
    Pre-emphasis filter:
        y[n] = x[n] - coeff * x[n-1]
    """
    return np.append(y[0], y[1:] - coeff * y[:-1])


def framing(y, sr, frame_size=0.025, frame_stride=0.010):
    """
    Split signal into overlapping short-time frames.
    Default:
        frame_size = 25 ms
        frame_stride = 10 ms
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
    Compute MFCC features manually:
    pre-emphasis -> framing -> windowing -> FFT -> mel filter bank
    -> log energy -> DCT -> first 13 coefficients.
    """
    emphasized = pre_emphasis(y, pre_emphasis_coeff)

    frames, frame_length, frame_step = framing(
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

    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_filter_banks = np.log(filter_banks)

    mfcc = dct(log_filter_banks, type=2, axis=1, norm="ortho")[:, :num_ceps]

    frame_times = np.arange(mfcc.shape[0]) * frame_step / sr

    return mfcc, frame_times


def plot_waveform(y, sr, output_path):
    t = np.arange(len(y)) / sr

    plt.figure(figsize=(8, 3))
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time-domain Waveform")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fft(y, sr, output_path):
    window = np.hamming(len(y))
    y_win = y * window

    spectrum = np.abs(rfft(y_win))
    freqs = rfftfreq(len(y_win), 1 / sr)

    spectrum = spectrum / np.max(spectrum)

    plt.figure(figsize=(8, 3))
    plt.plot(freqs, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.title("FFT Magnitude Spectrum")
    plt.xlim(0, 4000)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_spectrogram(y, sr, output_path):
    f, t, sxx = signal.spectrogram(
        y,
        fs=sr,
        window="hann",
        nperseg=512,
        noverlap=256,
        nfft=512,
        mode="magnitude",
    )

    sxx_db = 20 * np.log10(sxx / np.max(sxx) + np.finfo(float).eps)

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, sxx_db, shading="gouraud", vmin=-80, vmax=0)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram (STFT)")
    plt.ylim(0, 4000)
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_mfcc(mfcc, frame_times, output_path):
    plt.figure(figsize=(8, 4))

    if len(frame_times) > 1:
        extent = [frame_times[0], frame_times[-1], 1, mfcc.shape[1]]
    else:
        extent = [0, 0.01, 1, mfcc.shape[1]]

    plt.imshow(
        mfcc.T,
        origin="lower",
        aspect="auto",
        extent=extent,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficient Index")
    plt.title("MFCC Feature Map")
    plt.colorbar(label="Coefficient Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)

    y, sr = load_wav(SAMPLE_PATH)

    print(f"Loaded: {SAMPLE_PATH}")
    print(f"Sampling rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.3f} s")

    waveform_path = os.path.join(OUTPUT_DIR, "waveform.png")
    fft_path = os.path.join(OUTPUT_DIR, "fft_spectrum.png")
    spectrogram_path = os.path.join(OUTPUT_DIR, "spectrogram.png")
    mfcc_path = os.path.join(OUTPUT_DIR, "mfcc.png")

    print("Plotting waveform...")
    plot_waveform(y, sr, waveform_path)
    print(f"Saved: {waveform_path}")

    print("Plotting FFT...")
    plot_fft(y, sr, fft_path)
    print(f"Saved: {fft_path}")

    print("Plotting spectrogram...")
    plot_spectrogram(y, sr, spectrogram_path)
    print(f"Saved: {spectrogram_path}")

    print("Computing MFCC...")
    mfcc, frame_times = compute_mfcc(y, sr)

    print("Plotting MFCC...")
    plot_mfcc(mfcc[:, 1:], frame_times, mfcc_path)
    print(f"Saved: {mfcc_path}")

    print("Signal analysis finished.")


if __name__ == "__main__":
    main()