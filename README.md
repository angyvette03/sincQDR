# SincQDR-VAD

A Voice Activity Detection (VAD) system built with a custom `SincQDRVAD` neural network architecture. The model processes raw audio waveforms and outputs per-chunk speech probabilities, which are then merged into timestamped speech segments.

---

## Overview

SincQDR-VAD takes an audio file as input, slices it into overlapping 0.63-second windows, runs each window through a pretrained deep learning model, and outputs a list of time intervals where speech is detected.

Key features:
- Raw waveform input — no hand-crafted features required
- Sliding window chunking with configurable overlap
- Batch inference with sigmoid-based thresholding
- Contiguous speech segment merging
- Designed for deployment on low-resource hardware (e.g. Raspberry Pi)

---

## Repository Structure

```
sincQDR/
├── SincQDR-VAD/                        # Model architecture and training code
│   └── model/
│       └── sincqdrvad.py               # SincQDRVAD model definition
├── upload_to_pi.py                     # Inference script (run on Pi or workstation)
├── model_last_epoch.ckpt               # Latest model checkpoint
├── model_last_epoch_20260225_1.ckpt    # Dated checkpoint (Feb 2026)
├── model_last_epoch_29Jan.ckpt         # Dated checkpoint (Jan 2026)
└── .gitignore
```

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/)

Install dependencies:

```bash
pip install torch torchaudio
```

---

## Usage

### Running Inference

Edit the paths in `upload_to_pi.py` to point to your model checkpoint and audio file:

```python
wav_path = "/path/to/your/audio.wav"
checkpoint_path = "/path/to/model_last_epoch_20260225_1.ckpt"
```

Then run:

```bash
python upload_to_pi.py
```

Inference results are written to a log file (`pi_test5.log`) and printed to the console.

### Output

The script outputs:
- Per-chunk VAD probabilities (sigmoid of model logits)
- Merged speech segments as `(start_time, end_time)` tuples in seconds

Example log output:
```
VAD results (start, end, prob): [(0.0, 0.63, 0.91), (0.079, 0.71, 0.03), ...]
Merged speech segments: [(0.0, 1.26), (3.15, 5.04)]
```

---

## Configuration

Key parameters in `upload_to_pi.py`:

| Parameter        | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `PATCH_SIZE`     | `8`     | Patch size for the transformer/attention module  |
| `THRESHOLD`      | `0.5`   | Probability threshold for speech classification  |
| `sample_rate`    | `16000` | Expected audio sample rate (Hz)                  |
| `window_duration`| `0.63`  | Duration of each analysis window (seconds)       |
| `step_size`      | `~0.079`| Hop size between windows (12.5% of window)       |

The model is instantiated with:
```python
SincQDRVAD(in_channels=1, base_filters=32, out_filters=64, patch_size=8, num_classes=2, use_sinc=True)
```

---

## Model Checkpoints

Three checkpoints are included in the repository:

| File                                | Description               |
|-------------------------------------|---------------------------|
| `model_last_epoch.ckpt`             | Most recent training run  |
| `model_last_epoch_20260225_1.ckpt`  | Feb 25, 2026 checkpoint   |
| `model_last_epoch_29Jan.ckpt`       | Jan 29 checkpoint         |

Checkpoints store a `model_state` key compatible with `model.load_state_dict()`.

---

## Notes

- Audio is automatically resampled to 16 kHz if the source sample rate differs.
- Stereo audio is converted to mono by averaging channels before processing.
- Inference runs in batches of 8 chunks (matching the training batch size).
- The script is configured to run on CPU (`torch.device("cpu")`), making it suitable for edge deployment.
