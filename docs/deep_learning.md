# Deep Learning Notes

## Task

The deep learning part solves isolated digit recognition for digits 0-9. The training split is `speaker1` through `speaker5`; the validation split is `speaker6` and `speaker7`. Each speaker recording is segmented into ten utterances, assuming the digit order is `0,1,2,3,4,5,6,7,8,9`.

## Pipeline

1. Convert audio to mono 16 kHz waveform.
2. Segment each speaker file with an energy-based endpoint detector.
3. Add 100 ms margins around detected speech intervals.
4. Normalize each utterance, pad or truncate it to 1.5 s.
5. Optionally apply signal-system denoising:
   - `notch`: 50 Hz IIR notch filtering for power-line interference.
   - `spectral`: STFT magnitude spectral subtraction with noise estimated from the quietest frames.
   - `notch_spectral`: apply both.
6. Apply pre-emphasis with coefficient 0.97, matching the course document filter `[1, -0.97]`.
7. Apply 25 ms Hann windows, 10 ms hop, and FFT/STFT.
8. Extract the selected acoustic feature:
   - `mfcc` (default): Mel filter-bank log energy, DCT, first 13 cepstral coefficients, delta, and delta-delta.
   - `log_mel`: normalized log-Mel map for compatibility with earlier checkpoints.
9. Train a CNN classifier on the time-frequency feature maps.
10. Evaluate clean validation accuracy and confusion matrix.
11. Add mixed noise at the required SNR levels and plot accuracy versus SNR.

## Model

The main model is `TinyKeywordCNN` in `src/deep_learning/model.py`.

- Input: `1 x H x T` normalized feature map. With the default MFCC frontend, `H = 39` for 13 MFCC + delta + delta-delta.
- Convolution channels: `1 -> 16 -> 32 -> 64 -> 128`.
- Classifier: adaptive pooling, dropout 0.3, linear layer to 10 classes.
- Parameter count: about 102k trainable parameters.

This model size is intentional. The dataset has only about 50 training utterances after segmentation, so training a large neural network from scratch would overfit. A small CNN is easier to reproduce and easier to explain with signal-processing concepts: STFT, Mel filters, local time-frequency patterns, and convolution.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Inspect segmentation:

```bash
python -m src.deep_learning.segment
```

Generate the required signal-analysis figures for one digit:

```bash
python -m src.deep_learning.signal_analysis --speaker-id 1 --digit 0
```

Train:

```bash
python -m src.deep_learning.train
```

Evaluate clean validation data:

```bash
python -m src.deep_learning.evaluate
```

Evaluate mixed-noise robustness:

```bash
python -m src.deep_learning.noise_eval
```

Evaluate all generated noise types:

```bash
python -m src.deep_learning.noise_eval --noise-kinds mixed white pink power
```

Run the whole task workflow in one command:

```bash
python -m src.deep_learning.pipeline --epochs 120 --patience 20
```

Use the explicit signal-processing denoising frontend:

```bash
python -m src.deep_learning.pipeline --denoise notch_spectral --epochs 120 --patience 20
```

## Outputs

All generated experiment files are written to `outputs/deep_learning/`, which is ignored by Git:

- `signal_analysis/time_waveform.png`
- `signal_analysis/fft_magnitude.png`
- `signal_analysis/stft_spectrogram.png`
- `signal_analysis/log_mel_features.png`
- `signal_analysis/mfcc_features.png`
- `signal_analysis/signal_summary.json`
- `segments_manifest.json`
- `training_history.csv`
- `train_summary.json`
- `tiny_cnn.pt`
- `clean_metrics.json`
- `clean_confusion_matrix.csv`
- `clean_confusion_matrix.png`
- `noise_accuracy.csv`
- `noise_metrics.json`
- `accuracy_snr_curve.png`

## Report References

- Sainath and Parada, "Convolutional Neural Networks for Small-footprint Keyword Spotting", Interspeech 2015. This supports using compact CNNs for small-vocabulary speech recognition.
- Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting", arXiv:2106.04140. BC-ResNet reports strong Google Speech Commands V2 performance, but it targets a much larger dataset than this course task.
- Gong et al., "AST: Audio Spectrogram Transformer", and the `MIT/ast-finetuned-speech-commands-v2` model card. AST is a useful literature upper bound, but its 85M-parameter scale is too large for this small from-scratch experiment.

## Result Placeholders

After running the commands, copy the actual numbers from:

- Clean accuracy: `outputs/deep_learning/clean_metrics.json`
- Clean confusion matrix: `outputs/deep_learning/clean_confusion_matrix.png`
- SNR table: `outputs/deep_learning/noise_accuracy.csv`
- SNR curve: `outputs/deep_learning/accuracy_snr_curve.png`
