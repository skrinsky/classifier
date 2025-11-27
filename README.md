# Palette Classifier

A toolkit for **instrument-likeness classification** using a mel-spectrogram CNN
(`mel_v7` / `mel_v8`).  

It includes:

- A **Tkinter GUI** for auditioning audio and classifying clips.
- A **batch classifier** module for processing folders of WAVs.
- **Feature-building** scripts and **training** code for power users.

You can use this repo at three levels:

- **GUI only** – load in sounds, audition in original/pitch shifted +24/pitch shifted -24, and see top-3 predictions.
- **CLI / batch** – classify large folders of sounds into instrument-like classes.
- **Research / training** – modify datasets, rebuild mel caches, and retrain models.

---

## Repository Layout

```text
classifier/
├─ palette_gui.py          # Interactive GUI for audition + classification
│
├─ classify_palette/       # Batch classifier + model helper code
│  ├─ __init__.py
│  └─ classify_palette_octaves2.py
│
├─ best_models/            # Pretrained model checkpoints
│  ├─ best_model_v7.pt
│  └─ best_model_v8.pt
│
├─ label_mappings/         # Label ↔ index mappings for each model version
│  ├─ label_mapping_v7.json
│  └─ label_mapping_v8.json
│
├─ build_mels/             # Feature precomputation used during training
│  ├─ build_mels_v8.py
│  └─ build_mels_v9.py
│
├─ palette_csv/            # (Optional) example CSV outputs / your saved runs
│
├─ train/                  # (Optional) training scripts / notebooks
│
├─ README.md
└─ requirements.txt
Model / label versions
There are currently two model + label pairs:

v7

best_models/best_model_v7.pt

label_mappings/label_mapping_v7.json

v8

best_models/best_model_v8.pt

label_mappings/label_mapping_v8.json

You can load either pair in the GUI or on the command line.
By default, palette_gui.py is typically configured to auto-load the v7 pair
(see DEFAULT_MODEL_PATH and DEFAULT_LABELS_PATH at the top of the file), but
you can change that to v8 or just browse to whichever you want inside the GUI.

Installation
1. Clone the repo
bash
Copy code
git clone https://github.com/<your-username>/classifier.git
cd classifier
2. Create and activate a virtual environment
Using venv:

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
Or using conda:

bash
Copy code
conda create -n palette-gui python=3.10
conda activate palette-gui
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
torch from PyPI installs the CPU build by default.
If you need GPU acceleration, install the appropriate PyTorch wheel manually.

Running the GUI
From the repo root:

bash
Copy code
python palette_gui.py
What the GUI does
Loads a WAV file (mono-ized to 44.1 kHz internally).

Creates three variants:

orig – original audio

down2 – sampler-style pitch shift by −24 semitones

up2 – sampler-style pitch shift by +24 semitones

Shows a waveform for the current variant.

Lets you:

Click on the waveform to place the playback cursor.

Play from the cursor.

Optionally enable “follow cursor” so the red line tracks playback.

When playback reaches the end naturally, resets the cursor back to 0.

Runs the CNN on each variant and prints top-3 classes + confidences.

Model + labels in the GUI
On startup, the GUI will try to auto-load whatever paths are set as:

python
Copy code
DEFAULT_MODEL_PATH
DEFAULT_LABELS_PATH
in palette_gui.py, e.g.:

python
Copy code
DEFAULT_MODEL_PATH = BASE_DIR / "best_models" / "best_model_v7.pt"
DEFAULT_LABELS_PATH = BASE_DIR / "label_mappings" / "label_mapping_v7.json"
You can:

Leave those as-is (v7 default),

Change them to the v8 pair, or

Use the Browse buttons in the GUI to point to any .pt and matching
label_mapping_*.json file and click “Load Model + Labels”.

Batch Classification (CLI)
The batch classifier lives in:

text
Copy code
classify_palette/classify_palette_octaves2.py
You run it via the module interface so imports resolve correctly:

bash
Copy code
python -m classify_palette.classify_palette_octaves2 \
  --model best_models/best_model_v7.pt \
  --labels label_mappings/label_mapping_v7.json \
  --in_dir path/to/wavs \
  --out_csv palette_csv/palette_v7_predictions.csv
To use the v8 model instead:

bash
Copy code
python -m classify_palette.classify_palette_octaves2 \
  --model best_models/best_model_v8.pt \
  --labels label_mappings/label_mapping_v8.json \
  --in_dir path/to/wavs \
  --out_csv palette_csv/palette_v8_predictions.csv
Typical flags (see the script for the full set):

--model – path to a .pt checkpoint (v7 or v8).

--labels – matching label_mapping_*.json file.

--in_dir – directory containing WAV files.

--out_csv – where to write results.

--segment_seconds, --hop_seconds, --sr – advanced options controlling window size, hop, and sample rate.

Feature Building & Training (Advanced)
These are only needed if you want to retrain or deeply modify the model.

build_mels/
Scripts here precompute:

log-mel spectrograms

per-frame extras (low-band RMS, spectral rolloff, F0 contour)

global features (MFCC stats, spectral stats, chroma, F0 stats)

amplitude gating / segment selection

Example usage (exact flags may differ; check the script header):

bash
Copy code
python build_mels/build_mels_v9.py \
  --data_root /absolute/path/to/datasets \
  --out_dir /absolute/path/to/mel_cache
train/
This folder contains training scripts / notebooks used to:

Load the cached features,

Build and configure AudioCNNv6,

Train / fine-tune models that become best_model_v7.pt and best_model_v8.pt.

It’s included for reproducibility and experimentation, but isn’t required for
basic GUI or CLI usage.

Dependencies (human-readable)
Everything is listed in requirements.txt, but in summary:

numpy – array operations

torch – CNN model and inference

librosa – audio I/O, resampling, mel spectrograms, pitch shifting

soundfile – WAV reading/writing

sounddevice – playback from NumPy arrays

matplotlib – waveform visualization in the GUI

tkinter – built-in Python GUI toolkit (macOS / Linux / Windows)

Customization Notes
To change pitch-shift ranges (e.g. ±12 instead of ±24), edit the calls to
sampler_style_pitch_shift in palette_gui.py and/or
classify_palette_octaves2.py.

To plug in a new model version, just drop a new .pt into best_models/,
add a corresponding label_mapping_*.json into label_mappings/, and
point the GUI or CLI to that pair.

You can keep multiple versions around (v7, v8, etc.) and compare performance
simply by switching which pair you load.
