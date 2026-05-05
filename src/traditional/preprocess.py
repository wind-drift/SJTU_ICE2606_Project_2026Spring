import os
import csv
import numpy as np
import soundfile as sf
from pydub import AudioSegment


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SR = 16000

TRAIN_SPEAKERS = [1, 2, 3, 4, 5]
VAL_SPEAKERS = [6, 7]


def load_audio(path, target_sr=TARGET_SR):
    """
    Load mp3 audio with pydub, convert to mono, resample to target_sr,
    and normalize amplitude to [-1, 1].
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sr)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    max_abs = np.max(np.abs(samples))
    if max_abs > 0:
        samples = samples / max_abs

    return samples, target_sr


def compute_frame_rms(y, frame_length=1024, hop_length=256):
    """
    Compute short-time RMS energy.
    """
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    rms_values = []

    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start:start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values.append(rms)

    return np.array(rms_values)


def split_digits(
    y,
    sr,
    top_db=25,
    frame_length=1024,
    hop_length=256,
    min_duration=0.25,
    pad_duration=0.08,
):
    """
    Split one speaker audio into digit segments using simple short-time energy.
    """
    rms = compute_frame_rms(y, frame_length, hop_length)

    if len(rms) == 0 or np.max(rms) == 0:
        return []

    threshold = np.max(rms) * (10 ** (-top_db / 20))
    active = rms > threshold

    intervals = []
    in_segment = False
    start_frame = 0

    for i, is_active in enumerate(active):
        if is_active and not in_segment:
            in_segment = True
            start_frame = i

        elif not is_active and in_segment:
            in_segment = False
            end_frame = i

            start_sample = start_frame * hop_length
            end_sample = end_frame * hop_length + frame_length
            intervals.append((start_sample, end_sample))

    if in_segment:
        start_sample = start_frame * hop_length
        end_sample = len(y)
        intervals.append((start_sample, end_sample))

    segments = []
    pad = int(pad_duration * sr)

    for start, end in intervals:
        start = max(0, start - pad)
        end = min(len(y), end + pad)

        duration = (end - start) / sr
        if duration < min_duration:
            continue

        segment = y[start:end]
        segments.append(segment)

    return segments


def get_split_name(speaker_id):
    if speaker_id in TRAIN_SPEAKERS:
        return "train"
    elif speaker_id in VAL_SPEAKERS:
        return "val"
    else:
        raise ValueError(f"Unknown speaker id: {speaker_id}")


def prepare_dataset():
    metadata = []

    for speaker_id in TRAIN_SPEAKERS + VAL_SPEAKERS:
        filename = f"speaker{speaker_id}.mp3"
        input_path = os.path.join(RAW_DIR, filename)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Cannot find file: {input_path}")

        print(f"\nProcessing {filename}...")

        y, sr = load_audio(input_path)
        segments = split_digits(y, sr)

        print(f"Detected {len(segments)} segments.")

        if len(segments) != 10:
            raise RuntimeError(
                f"{filename} should contain 10 digits, "
                f"but detected {len(segments)} segments. "
                f"Please adjust segmentation parameters."
            )

        split_name = get_split_name(speaker_id)
        output_dir = os.path.join(PROCESSED_DIR, split_name)
        os.makedirs(output_dir, exist_ok=True)

        for label, segment in enumerate(segments):
            output_name = f"speaker{speaker_id}_digit{label}.wav"
            output_path = os.path.join(output_dir, output_name)

            sf.write(output_path, segment, sr)

            metadata.append({
                "path": output_path.replace("\\", "/"),
                "speaker": speaker_id,
                "label": label,
                "split": split_name,
            })

            print(f"Saved: {output_path}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    metadata_path = os.path.join(PROCESSED_DIR, "metadata.csv")

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "speaker", "label", "split"]
        )
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nMetadata saved to: {metadata_path}")
    print("Dataset preparation finished.")


if __name__ == "__main__":
    prepare_dataset()
