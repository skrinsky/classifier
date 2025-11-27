#!/usr/bin/env python3
"""
Precompute log-mel spectrograms + F0 contour + global features for instrument-likeness CNN.

For each dataset (Slakh, StemGMD single hits, IDMT drums, IDMT-SMT-BASS, IDMT-SMT-GUITAR,
Medley-solos), this script:
  - finds audio files that map into one of the CLASSES
  - extracts a SEGMENT_SECONDS-long mono segment at 44.1 kHz (pad if needed)
  - computes a log-mel spectrogram
  - computes an F0 contour aligned to mel frames using librosa.yin
  - computes a global feature vector (MFCCs, spectral stats, chroma, F0 stats)
  - saves the mel as .npy under out_root/<class>/ (shape: (n_mels, T))
  - saves the F0 contour as .npy under out_root/<class>/ (shape: (T,))
  - saves the feature vector as .npy under out_root/<class>/ (shape: (D,))
  - writes index.jsonl with entries:
      {
        "id":            unique_id,
        "dataset":       dataset_name,
        "class":         class_name,
        "class_idx":     int,
        "mel_path":      relative_path_to_mel_npy,
        "f0_path":       relative_path_to_f0_npy,
        "feat_path":     relative_path_to_feat_npy,
        "src_path":      original_audio_path,
        "median_f0_hz":  float,
        "voiced_frac":   float
      }

This is mel_data_v7: 
  - NO 'mallets' class at all (dropped from CLASSES and mappings).
  - NO 'strings' from Slakh (strings only from Medley-solos).
  - Bass only from IDMT-SMT-BASS.
  - Guitar only from IDMT-SMT-GUITAR + Medley guitar.
"""

# --- brutal coverage nuke: fake coverage + coverage.types completely ---
import sys
import types as _types

# Create a fake coverage.types module that happily returns a dummy for any attribute
cov_types = _types.ModuleType("coverage.types")


def _cov_types_getattr(name):
    # Any "from coverage.types import Whatever" or attribute access just gets a dummy
    return object


cov_types.__getattr__ = _cov_types_getattr

# Register this fake module so "import coverage.types" uses it
sys.modules["coverage.types"] = cov_types

# Create a fake top-level coverage module that exposes .types
cov_mod = _types.ModuleType("coverage")
cov_mod.types = cov_types
sys.modules["coverage"] = cov_mod

# Kill any active tracer just in case something turned coverage on
if sys.gettrace() is not None:
    sys.settrace(None)

# --- rest of your normal imports come AFTER this block ---
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, Optional, Iterable, Tuple

import numpy as np
import librosa
import soundfile as sf
import yaml

# ============================================================
# CONFIG
# ============================================================

BASE = Path("/scratch/summerk/320data")
# New folder for this run
OUT_ROOT = BASE / "mel_data_v7"

# Class set (NO MALLETS)
CLASSES = [
    "voice",
    "guitar",
    "bass",
    "piano",
    "synth",
    "strings",
    "winds_brass",
    "fx_noise",
    "kick",
    "snare",
    "hihat",
    "toms",
    "cymbals",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Audio / mel params
TARGET_SR = 44100
SEGMENT_SECONDS = 2.0  # 2-second segments
N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 128
FMIN = 20.0
FMAX = 8000.0  # instead of 20000.0

# F0 (YIN) params
F0_FMIN = 40.0     # Hz, lower pitch bound (E1-ish)
F0_FMAX = 1000.0   # Hz, upper pitch bound (covers voice/gtr, etc.)
F0_FRAME_LENGTH = N_FFT
F0_HOP_LENGTH = HOP_LENGTH  # align time steps to mel frames

# Skipping behavior
SKIP_EXISTING_MEL = True  # don't recompute mel if .npy already exists

# ============================================================
# MAPPING HELPERS
# ============================================================


def slakh_to_class(stem_info: Dict) -> Optional[str]:
    """
    Map Slakh stem metadata to one of our high-level classes.

    IMPORTANT for mel_data_v7:
      - NO Slakh stems mapped to 'bass' or 'guitar'.
      - NO Slakh stems mapped to 'strings'.
      - NO Slakh stems mapped to 'mallets'.
    Bass comes ONLY from IDMT-SMT-BASS.
    Guitar comes from IDMT-SMT-GUITAR (+ any Medley guitar).
    Strings come ONLY from Medley-solos.
    """
    inst_class = (stem_info.get("inst_class") or "").lower()
    inst_family = (stem_info.get("inst_family") or "").lower()
    is_drum = bool(stem_info.get("is_drum"))

    # Ignore Slakh drums; drum subclasses come from StemGMD/IDMT
    if is_drum:
        return None

    # DO NOT use Slakh strings at all in v7
    if "strings" in inst_class or "strings" in inst_family:
        return None

    # DO NOT use Slakh guitar in v7
    if "guitar" in inst_class or "guitar" in inst_family:
        return None

    # piano
    if "piano" in inst_class or "piano" in inst_family:
        return "piano"

    # winds / brass / reeds / pipes
    if any(k in inst_family for k in ["brass", "reed", "pipe"]):
        return "winds_brass"

    # mallets / chromatic percussion: DISABLED in v7
    if "chromatic percussion" in inst_class or "percussive" in inst_class:
        return None

    # sound effects / noise
    if inst_class in ["sound effects", "sound effect"] or "sound effects" in inst_class:
        return "fx_noise"

    # synth-ish (pads, leads, generic synth, organ)
    if any(
        k in inst_class
        for k in ["synth", "pad", "lead", "synthesizer", "synthesiser"]
    ) or "organ" in inst_class:
        return "synth"

    # vocals, if any appear as stems
    if "vocal" in inst_class or "voice" in inst_class:
        return "voice"

    return None


def classify_stemgmd(path: Path) -> Optional[str]:
    """
    Map StemGMD single-hit file path to drum subclass.
    Uses filename and parent folder name.
    """
    name = path.name.lower()
    parent_name = getattr(path.parent, "name", "")
    parent = str(parent_name).lower()
    text = f"{name} {parent}"

    if any(k in text for k in ["kick", "bd", "bassdrum", "bass_drum"]):
        return "kick"
    if any(k in text for k in ["snare", "sd"]):
        return "snare"
    if any(
        k in text for k in ["hihat", "hi-hat", "hh", "chh", "ohh", "closedhat", "openhat"]
    ):
        return "hihat"
    if any(k in text for k in ["tom", "t1", "t2", "t3", "t4"]):
        return "toms"
    if any(k in text for k in ["cym", "cymbal", "crash", "ride", "china", "splash"]):
        return "cymbals"
    return None


def classify_idmt(path: Path) -> Optional[str]:
    """
    Map IDMT-SMT-Drums file path to drum subclass.
    """
    name = path.name.lower()
    parent_name = getattr(path.parent, "name", "")
    parent = str(parent_name).lower()
    text = f"{name} {parent}"

    if any(k in text for k in ["kick", "kd", "bd", "bassdrum", "bass_drum"]):
        return "kick"
    if any(k in text for k in ["snare", "sd"]):
        return "snare"
    if any(
        k in text for k in ["hihat", "hi-hat", "hh", "chh", "ohh", "closedhat", "openhat"]
    ):
        return "hihat"
    return None


def medley_instrument_to_class(inst: str) -> Optional[str]:
    inst_lower = inst.strip().lower()

    # Voice
    if any(
        k in inst_lower
        for k in [
            "female singer",
            "male singer",
            "singing voice",
            "vocal",
            "singer",
            "voice",
        ]
    ):
        return "voice"

    # Guitar (Medley guitar is still allowed as guitar)
    if "guitar" in inst_lower:
        return "guitar"

    # Piano
    if "piano" in inst_lower:
        return "piano"

    # Strings (violin etc.) — THIS is our only source of 'strings' now
    if "violin" in inst_lower or "cello" in inst_lower or "viola" in inst_lower:
        return "strings"

    # Winds / brass
    if any(k in inst_lower for k in ["clarinet", "flute", "saxophone", "trumpet", "sax", "trombone"]):
        return "winds_brass"

    return None


# ============================================================
# AUDIO → MEL / F0 / FEATURES
# ============================================================


def load_segment(path: Path, sr: int, segment_seconds: float) -> np.ndarray:
    """
    Load a mono segment of fixed length (segment_seconds) at sample rate sr.

    Use soundfile.read instead of librosa.load to avoid environment issues
    with some files. Resample only if needed.
    """
    # Read with soundfile
    data, file_sr = sf.read(str(path), always_2d=False)

    # data can be shape (N,) or (N, C)
    if data.ndim == 2:
        # average channels to mono
        data = data.mean(axis=1)

    # Ensure float32
    data = data.astype(np.float32)

    # Resample if needed
    if file_sr != sr:
        try:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        except Exception as e:
            # If resample itself freaks out, just bail on this file
            raise RuntimeError(f"resample failed from {file_sr} -> {sr}: {e}")

    target_len = int(segment_seconds * sr)
    if data.shape[0] < target_len:
        data = np.pad(data, (0, target_len - data.shape[0]))
    elif data.shape[0] > target_len:
        data = data[:target_len]

    return data


def audio_to_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute log-mel spectrogram (dB) from mono audio.
    Shape: (n_mels, T_frames)
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)


def compute_f0_contour(y: np.ndarray, sr: int, T_target: int) -> Tuple[np.ndarray, float, float]:
    """
    Compute an F0 contour aligned (or very close) to mel frames.

    Returns:
      f0_norm:     np.ndarray of shape (T_target,), in [0, 1] (0 = unvoiced)
      median_hz:   float median fundamental (0.0 if no voiced frames)
      voiced_frac: float in [0, 1]
    """
    try:
        f0 = librosa.yin(
            y,
            fmin=F0_FMIN,
            fmax=F0_FMAX,
            sr=sr,
            frame_length=F0_FRAME_LENGTH,
            hop_length=F0_HOP_LENGTH,
        )
    except Exception as e:
        print(f"[WARN] YIN failed: {e}")
        f0 = np.zeros((T_target,), dtype=np.float32)

    f0 = np.asarray(f0, dtype=np.float32)

    # Treat non-positive or NaN as unvoiced
    good_mask = np.isfinite(f0) & (f0 > 0.0)
    voiced_frac = float(np.mean(good_mask)) if f0.size > 0 else 0.0

    if np.any(good_mask):
        median_f0_hz = float(np.median(f0[good_mask]))
        # Normalize F0 into [0, 1]; clip to bounds
        f0_clipped = np.clip(f0, F0_FMIN, F0_FMAX)
        f0_norm = (f0_clipped - F0_FMIN) / (F0_FMAX - F0_FMIN)
        f0_norm[~good_mask] = 0.0  # unvoiced -> 0
    else:
        median_f0_hz = 0.0
        f0_norm = np.zeros_like(f0, dtype=np.float32)

    # Align length with mel frames T_target (crop or pad with zeros)
    T_f0 = f0_norm.shape[0]
    if T_f0 > T_target:
        f0_norm = f0_norm[:T_target]
    elif T_f0 < T_target:
        pad = np.zeros((T_target - T_f0,), dtype=np.float32)
        f0_norm = np.concatenate([f0_norm, pad], axis=0)

    return f0_norm.astype(np.float32), median_f0_hz, voiced_frac


def compute_global_features(
    y: np.ndarray,
    sr: int,
    median_f0_hz: float,
    voiced_frac: float,
) -> np.ndarray:
    """
    Compute a global feature vector: MFCCs + spectral stats + chroma + F0 stats.

    Returns:
      feat_vec: 1D np.ndarray (D,)
    """
    feats = []

    # MFCCs
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feats.append(mfcc.mean(axis=1))
        feats.append(mfcc.std(axis=1))
    except Exception as e:
        print(f"[WARN] MFCC failed: {e}")
        feats.append(np.zeros(13, dtype=np.float32))
        feats.append(np.zeros(13, dtype=np.float32))

    # Spectral centroid / bandwidth / flatness
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats.append(cent.mean(axis=1))
        feats.append(cent.std(axis=1))
    except Exception as e:
        print(f"[WARN] spectral_centroid failed: {e}")
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    try:
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats.append(bw.mean(axis=1))
        feats.append(bw.std(axis=1))
    except Exception as e:
        print(f"[WARN] spectral_bandwidth failed: {e}")
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    try:
        flat = librosa.feature.spectral_flatness(y=y)
        feats.append(flat.mean(axis=1))
        feats.append(flat.std(axis=1))
    except Exception as e:
        print(f"[WARN] spectral_flatness failed: {e}")
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.append(contrast.mean(axis=1))
        feats.append(contrast.std(axis=1))
    except Exception as e:
        print(f"[WARN] spectral_contrast failed: {e}")
        feats.append(np.zeros(7, dtype=np.float32))
        feats.append(np.zeros(7, dtype=np.float32))

    # Chroma (rough pitch-class distribution)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        feats.append(chroma.mean(axis=1))
        feats.append(chroma.std(axis=1))
    except Exception as e:
        print(f"[WARN] chroma_cqt failed: {e}")
        feats.append(np.zeros(12, dtype=np.float32))
        feats.append(np.zeros(12, dtype=np.float32))

    # Zero-crossing rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        feats.append(zcr.mean(axis=1))
        feats.append(zcr.std(axis=1))
    except Exception as e:
        print(f"[WARN] zero_crossing_rate failed: {e}")
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    # F0 stats (normalized a bit)
    median_f0_norm = median_f0_hz / F0_FMAX if F0_FMAX > 0 else 0.0
    feats.append(np.array([median_f0_norm], dtype=np.float32))
    feats.append(np.array([voiced_frac], dtype=np.float32))

    feat_vec = np.concatenate(feats, axis=0).astype(np.float32)
    return feat_vec


def make_id(dataset: str, class_name: str, audio_path: Path) -> str:
    """
    Make a (roughly) unique id string from dataset, class, and relative path.
    """
    rel = str(audio_path.relative_to(BASE))
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return f"{dataset}_{class_name}_{h}"


# ============================================================
# DATASET ITERATORS
# yield (dataset_name, class_name, Path_to_audio)
# ============================================================


def iter_slakh_audio() -> Iterable[Tuple[str, str, Path]]:
    """
    Yield (dataset, class, audio_path) for Slakh stems that map into our classes.

    We *do not* trust audio_rendered/audio_degraded fields (they can be bools);
    instead, we assume the standard layout: stems/<stem_id>.flac

    NOTE (v7): This iterator will NEVER produce 'bass', 'guitar', 'strings', or 'mallets'.
    """
    slakh_root = BASE / "slakh2100" / "slakh2100_flac_redux"
    meta_paths = list(slakh_root.rglob("metadata.yaml"))
    print(f"[Slakh] Found {len(meta_paths)} metadata.yaml files under {slakh_root}")

    for meta_path in meta_paths:
        track_dir = meta_path.parent
        with meta_path.open("r") as f:
            meta = yaml.safe_load(f)
        stems = meta.get("stems", {})

        for stem_id, stem_info in stems.items():
            lab = slakh_to_class(stem_info)
            if lab is None or lab not in CLASS_TO_IDX:
                continue

            # Standard Slakh layout: stems/<stem_id>.flac
            audio_path = track_dir / "stems" / f"{stem_id}.flac"
            if not audio_path.exists():
                # If this assumption fails for some tracks, just skip them
                continue

            yield ("slakh2100", lab, audio_path)


def iter_stemgmd_audio() -> Iterable[Tuple[str, str, Path]]:
    root = BASE / "stemgmd" / "single_hits"
    wavs = list(root.rglob("*.wav"))
    print(f"[StemGMD] Found {len(wavs)} .wav files under {root}")
    for w in wavs:
        lab = classify_stemgmd(w)
        if lab is None or lab not in CLASS_TO_IDX:
            continue
        yield ("stemgmd", lab, w)


def iter_idmt_audio() -> Iterable[Tuple[str, str, Path]]:
    """
    IDMT-SMT-Drums (kick, snare, hihat only).
    """
    root = BASE / "idmt_smt_drums" / "audio"
    wavs = list(root.rglob("*.wav"))
    print(f"[IDMT-Drums] Found {len(wavs)} .wav files under {root}")
    for w in wavs:
        lab = classify_idmt(w)
        if lab is None or lab not in CLASS_TO_IDX:
            continue
        yield ("idmt_smt_drums", lab, w)


def iter_idmt_bass_audio() -> Iterable[Tuple[str, str, Path]]:
    """
    IDMT-SMT-BASS: ALL .wav files are treated as 'bass' class.

    This is the ONLY source of 'bass' in this experiment.
    """
    root = BASE / "idmt_smt_bass" / "IDMT-SMT-BASS"
    wavs = list(root.rglob("*.wav"))
    print(f"[IDMT-Bass] Found {len(wavs)} .wav files under {root}")
    for w in wavs:
        yield ("idmt_smt_bass", "bass", w)


def iter_idmt_guitar_audio() -> Iterable[Tuple[str, str, Path]]:
    """
    IDMT-SMT-GUITAR_V2: ALL .wav files are treated as 'guitar' class.

    This is the primary source of 'guitar' in this experiment.
    We assume the dataset was unzipped under BASE / 'idmt_smt_guitar'.
    """
    root = BASE / "idmt_smt_guitar"
    wavs = list(root.rglob("*.wav"))
    print(f"[IDMT-Guitar] Found {len(wavs)} .wav files under {root}")
    for w in wavs:
        yield ("idmt_smt_guitar", "guitar", w)


def _build_medley_uuid_to_instrument(meta_csv: Path) -> Dict[str, str]:
    """
    Read Medley-solos metadata and return a dict: uuid4 -> instrument string.

    We strip whitespace from column names so that e.g. ' uuid4 ' becomes 'uuid4'.
    """
    uuid2inst: Dict[str, str] = {}
    with meta_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            # Normalize keys: strip whitespace
            row = {(k.strip() if k is not None else ""): v for k, v in raw_row.items()}

            subset = (row.get("subset") or "").strip()
            inst = (row.get("instrument") or "").strip()
            uuid = (row.get("uuid4") or row.get("UUID4") or "").strip()

            if not subset or not inst or not uuid:
                continue

            uuid2inst[uuid] = inst
    return uuid2inst


def iter_medley_audio() -> Iterable[Tuple[str, str, Path]]:
    """
    Yield (dataset, class, audio_path) from Medley-solos.

    In v7 this is the ONLY source of 'strings'.
    """
    medley_root = BASE / "medley_solos"
    meta_csv = medley_root / "Medley-solos-DB_metadata.csv"
    if not meta_csv.exists():
        print(f"[Medley] Metadata CSV not found at {meta_csv}")
        return

    uuid2inst = _build_medley_uuid_to_instrument(meta_csv)
    if not uuid2inst:
        print("[Medley] Warning: uuid->instrument map is empty; no examples will be yielded.")
        return

    # Skip macOS resource-fork files like "._Medley-solos-DB_..."
    wavs = [
        w
        for w in medley_root.rglob("*.wav")
        if not w.name.startswith("._")
    ]

    print(f"[Medley] Found {len(wavs)} .wav files under {medley_root} (after skipping '._*')")

    for w in wavs:
        stem = w.stem
        parts = stem.split("_")
        if not parts:
            continue
        uuid = parts[-1]
        inst = uuid2inst.get(uuid)
        if not inst:
            continue

        lab = medley_instrument_to_class(inst)
        if lab is None or lab not in CLASS_TO_IDX:
            continue

        yield ("medley_solos", lab, w)


# ============================================================
# MAIN PRECOMPUTATION
# ============================================================


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = OUT_ROOT / "index.jsonl"

    # NOTE:
    # - Slakh produces NO 'bass', 'guitar', 'strings', or 'mallets' in v7.
    # - Bass comes only from idmt_smt_bass.
    # - Guitar comes primarily from idmt_smt_guitar (+ any Medley guitar).
    # - Strings come ONLY from Medley-solos.
    datasets = [
        ("slakh2100",        iter_slakh_audio),
        ("stemgmd",          iter_stemgmd_audio),
        ("idmt_smt_drums",   iter_idmt_audio),
        ("idmt_smt_bass",    iter_idmt_bass_audio),
        ("idmt_smt_guitar",  iter_idmt_guitar_audio),
        ("medley_solos",     iter_medley_audio),
    ]

    num_written = 0
    dataset_counts: Dict[str, int] = {}
    dataset_class_counts: Dict[str, Dict[str, int]] = {}

    with index_path.open("w") as index_f:
        for ds_name, iter_fn in datasets:
            print(f"\n=== Processing dataset: {ds_name} ===")
            for dataset, class_name, audio_path in iter_fn():
                class_idx = CLASS_TO_IDX[class_name]

                # Output dirs
                class_dir = OUT_ROOT / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                uid = make_id(dataset, class_name, audio_path)
                mel_rel = Path(class_name) / f"{uid}.npy"
                mel_path = OUT_ROOT / mel_rel
                f0_rel = Path(class_name) / f"{uid}_f0.npy"
                f0_path = OUT_ROOT / f0_rel
                feat_rel = Path(class_name) / f"{uid}_feat.npy"
                feat_path = OUT_ROOT / feat_rel

                # Load segment once (for mel, F0, and features)
                try:
                    y = load_segment(
                        audio_path, sr=TARGET_SR, segment_seconds=SEGMENT_SECONDS
                    )
                except Exception as e:
                    print(f"[WARN] Failed to load {audio_path}: {e}")
                    continue

                # Compute or reuse mel
                if SKIP_EXISTING_MEL and mel_path.exists():
                    try:
                        mel = np.load(mel_path).astype(np.float32)
                    except Exception as e:
                        print(f"[WARN] Failed to reload existing mel {mel_path}: {e}")
                        # fall back to recompute
                        try:
                            mel = audio_to_logmel(y, sr=TARGET_SR)
                            np.save(mel_path, mel)
                        except Exception as e2:
                            print(f"[WARN] Failed mel recompute for {audio_path}: {e2}")
                            continue
                else:
                    try:
                        mel = audio_to_logmel(y, sr=TARGET_SR)
                        np.save(mel_path, mel)
                    except Exception as e:
                        print(f"[WARN] Failed mel for {audio_path}: {e}")
                        continue

                # Compute F0 contour aligned to mel frames
                _, T_mel = mel.shape
                f0_norm, median_f0_hz, voiced_frac = compute_f0_contour(
                    y, sr=TARGET_SR, T_target=T_mel
                )
                try:
                    np.save(f0_path, f0_norm)
                except Exception as e:
                    print(f"[WARN] Failed saving F0 for {audio_path}: {e}")
                    continue

                # Compute global feature vector (MFCCs + spectral + chroma + F0 stats)
                try:
                    feat_vec = compute_global_features(
                        y=y,
                        sr=TARGET_SR,
                        median_f0_hz=median_f0_hz,
                        voiced_frac=voiced_frac,
                    )
                    np.save(feat_path, feat_vec)
                except Exception as e:
                    print(f"[WARN] Failed global features for {audio_path}: {e}")
                    continue

                # Write index record
                rec = {
                    "id": uid,
                    "dataset": dataset,
                    "class": class_name,
                    "class_idx": class_idx,
                    "mel_path": str(mel_rel),
                    "f0_path": str(f0_rel),
                    "feat_path": str(feat_rel),
                    "src_path": str(audio_path),
                    "median_f0_hz": float(median_f0_hz),
                    "voiced_frac": float(voiced_frac),
                }
                index_f.write(json.dumps(rec) + "\n")
                num_written += 1

                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
                ds_cc = dataset_class_counts.setdefault(dataset, {})
                ds_cc[class_name] = ds_cc.get(class_name, 0) + 1

                if num_written % 100 == 0:
                    print(f"  wrote {num_written} mel+f0+feat examples so far...")

    print(f"\nDone. Wrote index with {num_written} entries to {index_path}")

    # Summary: per dataset
    print("\n=== EXAMPLES PER DATASET ===")
    for ds_name, _ in datasets:
        count = dataset_counts.get(ds_name, 0)
        print(f"{ds_name:14s}: {count}")

    # Summary: per dataset + class
    print("\n=== EXAMPLES PER DATASET + CLASS ===")
    for ds_name, _ in datasets:
        if ds_name not in dataset_class_counts:
            continue
        print(f"\n[{ds_name}]")
        class_counts = dataset_class_counts[ds_name]
        for cls in sorted(class_counts.keys()):
            print(f"  {cls:10s}: {class_counts[cls]}")


if __name__ == "__main__":
    main()
