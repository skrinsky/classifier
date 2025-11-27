#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-classify audio files into instrument-like categories using the trained CNN.

Defaults (you can edit these or override via CLI):
  - Checkpoint:     ~/Documents/GitHub/classifier/best_model.pt
  - Label mapping:  ~/Documents/GitHub/classifier/label_mapping.json
  - Audio dir:      /Users/summerkrinsky/Downloads/summer 2/palette/palette/Bounces/
  - Output CSV:     palette_predictions.csv

Each .wav file is:
  - loaded mono at 44.1 kHz
  - split into 2-second chunks with 50% overlap
  - each chunk -> mel + time-varying extras (low-band energy, rolloff, F0)
                  + global MFCC/spectral/chroma/F0 stats
  - model logits are averaged over chunks, then softmaxed to get per-class probs.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

# ====== AUDIO / MEL CONFIG (must match training) ======
TARGET_SR = 44100
SEGMENT_SECONDS = 2.0
SEGMENT_SAMPLES = int(TARGET_SR * SEGMENT_SECONDS)

N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 128
FMIN = 20.0
FMAX = 8000.0

# F0 (YIN) params
F0_FMIN = 40.0
F0_FMAX = 1000.0
F0_FRAME_LENGTH = N_FFT
F0_HOP_LENGTH = HOP_LENGTH


# ====== MODEL DEFINITION (same as training) ======

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding="same")
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d((2, 2)) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class AudioCNNv6(nn.Module):
    def __init__(self, n_classes, extra_dim=0, global_dim=0):
        super().__init__()

        self.mel_branch = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),
        )
        self.gap2d = nn.AdaptiveAvgPool2d((1, 1))

        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.feat_conv = nn.Conv1d(extra_dim, 32, kernel_size=3, padding=1)
            self.gap1d = nn.AdaptiveAvgPool1d(1)
            extra_out = 32
        else:
            self.feat_conv = None
            self.gap1d = None
            extra_out = 0

        self.global_dim = global_dim
        fc_in = 256 + extra_out + global_dim
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, mel, extra_feats=None, global_feats=None):
        # mel: (B, 1, M_mel, T)
        xm = self.mel_branch(mel)
        xm = self.gap2d(xm)
        xm = xm.view(xm.size(0), -1)

        pieces = [xm]

        if self.extra_dim > 0 and extra_feats is not None:
            f = self.feat_conv(extra_feats)        # (B, 32, T)
            f = self.gap1d(f).squeeze(-1)          # (B, 32)
            pieces.append(f)

        if self.global_dim > 0 and global_feats is not None:
            pieces.append(global_feats)

        x = torch.cat(pieces, dim=1)
        logits = self.fc(x)
        return logits


# ====== FEATURE HELPERS (mirror preprocessing) ======

def load_audio(path, sr=TARGET_SR):
    """Load mono audio, resampling if needed."""
    data, file_sr = sf.read(str(path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)

    return data


def chunk_audio(y, segment_samples=SEGMENT_SAMPLES, hop_samples=None):
    """Yield overlapping chunks of audio of fixed length."""
    if hop_samples is None:
        hop_samples = segment_samples // 2  # 50% overlap

    n = len(y)
    if n <= segment_samples:
        # single padded chunk
        out = np.zeros(segment_samples, dtype=np.float32)
        out[:n] = y
        yield out
        return

    start = 0
    while start < n:
        end = start + segment_samples
        chunk = y[start:end]
        if len(chunk) < segment_samples:
            pad = np.zeros(segment_samples, dtype=np.float32)
            pad[:len(chunk)] = chunk
            chunk = pad
        yield chunk
        start += hop_samples


def audio_to_logmel(y, sr=TARGET_SR):
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


def compute_f0_contour(y, sr, T_target):
    try:
        f0 = librosa.yin(
            y,
            fmin=F0_FMIN,
            fmax=F0_FMAX,
            sr=sr,
            frame_length=F0_FRAME_LENGTH,
            hop_length=F0_HOP_LENGTH,
        )
    except Exception:
        f0 = np.zeros((T_target,), dtype=np.float32)

    f0 = np.asarray(f0, dtype=np.float32)
    good_mask = np.isfinite(f0) & (f0 > 0.0)
    voiced_frac = float(np.mean(good_mask)) if f0.size > 0 else 0.0

    if np.any(good_mask):
        median_f0_hz = float(np.median(f0[good_mask]))
        f0_clipped = np.clip(f0, F0_FMIN, F0_FMAX)
        f0_norm = (f0_clipped - F0_FMIN) / (F0_FMAX - F0_FMIN)
        f0_norm[~good_mask] = 0.0
    else:
        median_f0_hz = 0.0
        f0_norm = np.zeros_like(f0, dtype=np.float32)

    T_f0 = f0_norm.shape[0]
    if T_f0 > T_target:
        f0_norm = f0_norm[:T_target]
    elif T_f0 < T_target:
        pad = np.zeros((T_target - T_f0,), dtype=np.float32)
        f0_norm = np.concatenate([f0_norm, pad], axis=0)

    return f0_norm.astype(np.float32), median_f0_hz, voiced_frac


def compute_global_features(y, sr, median_f0_hz, voiced_frac):
    feats = []

    # MFCCs (13)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feats.append(mfcc.mean(axis=1))
        feats.append(mfcc.std(axis=1))
    except Exception:
        feats.append(np.zeros(13, dtype=np.float32))
        feats.append(np.zeros(13, dtype=np.float32))

    # spectral centroid
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats.append(cent.mean(axis=1))
        feats.append(cent.std(axis=1))
    except Exception:
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    # spectral bandwidth
    try:
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats.append(bw.mean(axis=1))
        feats.append(bw.std(axis=1))
    except Exception:
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    # spectral flatness
    try:
        flat = librosa.feature.spectral_flatness(y=y)
        feats.append(flat.mean(axis=1))
        feats.append(flat.std(axis=1))
    except Exception:
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    # spectral contrast (7 bands)
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.append(contrast.mean(axis=1))
        feats.append(contrast.std(axis=1))
    except Exception:
        feats.append(np.zeros(7, dtype=np.float32))
        feats.append(np.zeros(7, dtype=np.float32))

    # chroma (12)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        feats.append(chroma.mean(axis=1))
        feats.append(chroma.std(axis=1))
    except Exception:
        feats.append(np.zeros(12, dtype=np.float32))
        feats.append(np.zeros(12, dtype=np.float32))

    # zero-crossing rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        feats.append(zcr.mean(axis=1))
        feats.append(zcr.std(axis=1))
    except Exception:
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    # F0 stats
    median_f0_norm = median_f0_hz / F0_FMAX if F0_FMAX > 0 else 0.0
    feats.append(np.array([median_f0_norm], dtype=np.float32))
    feats.append(np.array([voiced_frac], dtype=np.float32))

    feat_vec = np.concatenate(feats, axis=0).astype(np.float32)
    return feat_vec


def prepare_features_for_chunk(chunk, sr=TARGET_SR):
    """
    Given a 1D audio chunk (length SEGMENT_SAMPLES), compute:
      - mel tensor:    (1, 1, n_mels, T)
      - extras tensor: (1, 3, T)
      - global tensor: (1, D)
    """
    mel_db = audio_to_logmel(chunk, sr=sr)            # (M, T)
    M_mel, T = mel_db.shape

    # per-example normalization (same as training)
    mel = mel_db.copy()
    mean = mel.mean()
    std = mel.std()
    if std < 1e-6:
        std = 1e-6
    mel = (mel - mean) / std
    mel = np.expand_dims(mel, axis=0)                 # (1, M, T)

    # time-varying extras
    power = np.power(10.0, mel_db / 10.0, dtype=np.float32)
    LOW_BINS = min(8, M_mel)
    low_band = power[:LOW_BINS, :]
    low_energy = low_band.mean(axis=0, keepdims=True)     # (1, T)

    spec_total = power.sum(axis=0, keepdims=True) + 1e-10
    cum = np.cumsum(power, axis=0)
    thresh = 0.85 * spec_total
    rolloff_bin = (cum < thresh).sum(axis=0)              # (T,)
    rolloff_norm = rolloff_bin.astype(np.float32) / max(M_mel - 1, 1)
    rolloff_row = rolloff_norm.reshape(1, T)              # (1, T)

    mean_total_energy = spec_total.mean(axis=0, keepdims=True)
    low_energy_norm = low_energy / (mean_total_energy + 1e-10)

    f0_norm, median_f0_hz, voiced_frac = compute_f0_contour(chunk, sr=sr, T_target=T)
    f0_row = f0_norm.reshape(1, T)

    extra_feats = np.vstack([low_energy_norm, rolloff_row, f0_row]).astype(np.float32)  # (3, T)

    global_vec = compute_global_features(
        y=chunk, sr=sr, median_f0_hz=median_f0_hz, voiced_frac=voiced_frac
    )

    # to tensors with batch dim
    mel_t = torch.from_numpy(mel).unsqueeze(0)         # (1, 1, M, T)
    extra_t = torch.from_numpy(extra_feats).unsqueeze(0)  # (1, 3, T)
    global_t = torch.from_numpy(global_vec).unsqueeze(0)  # (1, D)

    return mel_t.float(), extra_t.float(), global_t.float()


# ====== MAIN CLASSIFICATION LOGIC ======

def classify_directory(args):
    ckpt_path = os.path.expanduser(args.ckpt)
    mapping_path = os.path.expanduser(args.label_mapping)
    audio_dir = Path(args.audio_dir)
    out_csv_path = Path(args.out_csv)

    assert audio_dir.is_dir(), f"Audio dir not found: {audio_dir}"

    # load label mapping
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    idx_to_label = mapping.get("idx_to_label", None)
    if idx_to_label is None:
        # mapping stored as {label_to_idx, idx_to_label}, fallback if not
        idx_to_label = {int(v): k for k, v in mapping["label_to_idx"].items()}
    # make sure keys are ints
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    n_classes = len(idx_to_label)

    # collect wavs
    wav_paths = sorted(p for p in audio_dir.rglob("*.wav"))
    if not wav_paths:
        print(f"No .wav files found under {audio_dir}")
        return

    print(f"Found {len(wav_paths)} .wav files under {audio_dir}")

    # we need global_dim for model; compute from first chunk of first file
    first_y = load_audio(wav_paths[0])
    first_chunk = next(chunk_audio(first_y))
    mel_t, extra_t, global_t = prepare_features_for_chunk(first_chunk)
    extra_dim = extra_t.shape[1]
    global_dim = global_t.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model = AudioCNNv6(n_classes=n_classes, extra_dim=extra_dim, global_dim=global_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    results = []

    for wav_path in wav_paths:
        y = load_audio(wav_path)
        all_logits = []

        with torch.no_grad():
            for chunk in chunk_audio(y):
                mel_t, extra_t, global_t = prepare_features_for_chunk(chunk)
                mel_t = mel_t.to(device)
                extra_t = extra_t.to(device)
                global_t = global_t.to(device)

                logits = model(mel_t, extra_t, global_t)  # (1, C)
                all_logits.append(logits.cpu())

        if not all_logits:
            continue

        stacked = torch.cat(all_logits, dim=0)      # (N_chunks, C)
        mean_logits = stacked.mean(dim=0, keepdim=True)
        probs = F.softmax(mean_logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = probs.argsort()[::-1][:3]
        top_labels = [idx_to_label[int(i)] for i in top_indices]
        top_probs = [float(probs[int(i)]) for i in top_indices]

        primary_label = top_labels[0]
        primary_conf = top_probs[0]

        rel_name = wav_path.name
        print(f"{rel_name}: {primary_label} ({primary_conf:.3f})  "
              f"| top3: " +
              ", ".join(f"{lab} {p:.3f}" for lab, p in zip(top_labels, top_probs)))

        results.append({
            "file": str(wav_path),
            "primary_label": primary_label,
            "primary_conf": primary_conf,
            "top1_label": top_labels[0],
            "top1_prob": top_probs[0],
            "top2_label": top_labels[1],
            "top2_prob": top_probs[1],
            "top3_label": top_labels[2],
            "top3_prob": top_probs[2],
        })

    # write CSV
    with out_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "primary_label",
                "primary_conf",
                "top1_label",
                "top1_prob",
                "top2_label",
                "top2_prob",
                "top3_label",
                "top3_prob",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSaved predictions to {out_csv_path}")


def get_args():
    ap = argparse.ArgumentParser("Batch classifier for field recordings using mel_v7 CNN.")
    ap.add_argument(
        "--ckpt",
        default="~/Documents/GitHub/classifier/best_model.pt",
        help="Path to trained model checkpoint (.pt).",
    )
    ap.add_argument(
        "--label_mapping",
        default="~/Documents/GitHub/classifier/label_mapping.json",
        help="Path to label_mapping.json saved during training.",
    )
    ap.add_argument(
        "--audio_dir",
        default="/Users/summerkrinsky/Downloads/summer 2/palette/palette/Bounces/",
        help="Directory containing .wav files to classify.",
    )
    ap.add_argument(
        "--out_csv",
        default="palette_predictions.csv",
        help="Where to write CSV with predictions (relative or absolute).",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    classify_directory(args)
