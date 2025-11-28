#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classify arbitrary audio recordings into instrument-like categories
using the mel_v7 CNN, plus +/- 2 octave pitch-shifted variants.

For each .wav file in --in_dir, we:
  - load and resample to 44.1 kHz mono
  - create three versions:
        'orig'  : original
        'down2' : shifted -24 semitones (sampler-style rate change)
        'up2'   : shifted +24 semitones (sampler-style rate change)
    (pitch shifting is done by resampling, like a classic sampler:
     changing playback rate so pitch and duration are linked)
  - chop into 2-second chunks with 1-second hop
  - compute log-mel + time-varying extras + global features
  - run the CNN on each chunk and average logits over time
  - softmax to get class probabilities
  - record the top-3 predicted classes for each variant

Outputs a CSV with columns:
  file, variant, primary_label, primary_conf,
  top1, p1, top2, p2, top3, p3
"""

import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- CONSTANTS (match mel_data_v7) -----------------

TARGET_SR = 44100
SEGMENT_SECONDS = 2.0
HOP_SECONDS = 1.0  # 50% overlap for 2s windows

N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 128
FMIN = 20.0
FMAX = 8000.0

F0_FMIN = 40.0
F0_FMAX = 1000.0
F0_FRAME_LENGTH = N_FFT
F0_HOP_LENGTH = HOP_LENGTH


# ----------------- MODEL ARCH (AudioCNNv6) -----------------


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
    def __init__(self, n_classes, extra_dim=3, global_dim=0):
        """
        n_classes: number of instrument classes
        extra_dim: number of time-varying extra channels (3: low_energy, rolloff, f0)
        global_dim: dimensionality of the global feature vector
        """
        super().__init__()

        # 2D CNN over mel
        self.mel_branch = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),  # then global avg pool
        )
        self.gap2d = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 256, 1, 1)

        # 1D conv over time-varying extras
        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.feat_conv = nn.Conv1d(extra_dim, 32, kernel_size=3, padding=1)
            self.gap1d = nn.AdaptiveAvgPool1d(1)  # -> (B, 32, 1)
            extra_out = 32
        else:
            self.feat_conv = None
            self.gap1d = None
            extra_out = 0

        self.global_dim = global_dim

        fc_in = 256 + extra_out + global_dim
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, mel, extra_feats=None, global_feats=None):
        """
        mel:          (B, 1, n_mels, T)
        extra_feats:  (B, extra_dim, T) or None
        global_feats: (B, global_dim) or None
        """
        # mel branch
        xm = self.mel_branch(mel)        # (B, 256, H', W')
        xm = self.gap2d(xm)              # (B, 256, 1, 1)
        xm = xm.view(xm.size(0), -1)     # (B, 256)

        pieces = [xm]

        if self.extra_dim > 0 and extra_feats is not None:
            f = self.feat_conv(extra_feats)     # (B, 32, T)
            f = self.gap1d(f).squeeze(-1)       # (B, 32)
            pieces.append(f)

        if self.global_dim > 0 and global_feats is not None:
            pieces.append(global_feats)

        x = torch.cat(pieces, dim=1)
        logits = self.fc(x)
        return logits


# ----------------- FEATURE HELPERS -----------------


def load_audio_mono(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load file at native SR then resample to target_sr mono float32."""
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data


def chunk_signal(y: np.ndarray, sr: int, seg_seconds: float, hop_seconds: float):
    """Yield overlapping chunks of y, each seg_seconds long."""
    seg_len = int(seg_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))

    start = 0
    while start + seg_len <= len(y):
        yield y[start:start + seg_len]
        start += hop_len


def audio_to_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute log-mel spectrogram (dB). Returns (n_mels, T)."""
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


def compute_f0_contour(y: np.ndarray, sr: int, T_target: int):
    """Compute normalized F0 contour aligned to mel frames."""
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

    # align to T_target
    T_f0 = f0_norm.shape[0]
    if T_f0 > T_target:
        f0_norm = f0_norm[:T_target]
    elif T_f0 < T_target:
        pad = np.zeros((T_target - T_f0,), dtype=np.float32)
        f0_norm = np.concatenate([f0_norm, pad], axis=0)

    return f0_norm.astype(np.float32), median_f0_hz, voiced_frac


def compute_global_features(y: np.ndarray, sr: int,
                            median_f0_hz: float,
                            voiced_frac: float) -> np.ndarray:
    """Global MFCC + spectral + chroma + F0 stats (same as training)."""
    feats = []

    # MFCCs
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feats.append(mfcc.mean(axis=1))
        feats.append(mfcc.std(axis=1))
    except Exception:
        feats.append(np.zeros(13, dtype=np.float32))
        feats.append(np.zeros(13, dtype=np.float32))

    # Spectral centroid / bandwidth / flatness
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats.append(cent.mean(axis=1))
        feats.append(cent.std(axis=1))
    except Exception:
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    try:
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats.append(bw.mean(axis=1))
        feats.append(bw.std(axis=1))
    except Exception:
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    try:
        flat = librosa.feature.spectral_flatness(y=y)
        feats.append(flat.mean(axis=1))
        feats.append(flat.std(axis=1))
    except Exception:
        feats.append(np.zeros(1, dtype=np.float32))
        feats.append(np.zeros(1, dtype=np.float32))

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.append(contrast.mean(axis=1))
        feats.append(contrast.std(axis=1))
    except Exception:
        feats.append(np.zeros(7, dtype=np.float32))
        feats.append(np.zeros(7, dtype=np.float32))

    # Chroma
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        feats.append(chroma.mean(axis=1))
        feats.append(chroma.std(axis=1))
    except Exception:
        feats.append(np.zeros(12, dtype=np.float32))
        feats.append(np.zeros(12, dtype=np.float32))

    # Zero-crossing
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


def chunk_to_model_inputs(chunk: np.ndarray):
    """
    Given a 2s chunk (float32, 44.1k), compute:
      mel_norm: (1, 1, n_mels, T)
      extra_feats: (1, 3, T)
      global_feats: (1, D)
    """
    # mel
    mel_db = audio_to_logmel(chunk, TARGET_SR)  # (M, T)
    M_mel, T = mel_db.shape

    # per-example normalization (like training)
    mel = mel_db.copy()
    mean = mel.mean()
    std = mel.std()
    if std < 1e-6:
        std = 1e-6
    mel = (mel - mean) / std
    mel = mel[np.newaxis, np.newaxis, :, :]  # (1,1,M,T)

    # time-varying extras
    power = np.power(10.0, mel_db / 10.0, dtype=np.float32)  # (M, T)
    LOW_BINS = min(8, M_mel)
    low_band = power[:LOW_BINS, :]
    low_energy = low_band.mean(axis=0, keepdims=True)  # (1, T)

    spec_total = power.sum(axis=0, keepdims=True) + 1e-10
    cum = np.cumsum(power, axis=0)
    thresh = 0.85 * spec_total
    rolloff_bin = (cum < thresh).sum(axis=0)
    rolloff_norm = rolloff_bin.astype(np.float32) / max(M_mel - 1, 1)
    rolloff_row = rolloff_norm.reshape(1, T)

    mean_total_energy = spec_total.mean(axis=0, keepdims=True)
    low_energy_norm = low_energy / (mean_total_energy + 1e-10)

    f0_norm, median_f0_hz, voiced_frac = compute_f0_contour(chunk, TARGET_SR, T)
    f0_row = f0_norm.reshape(1, T)

    extra_feats = np.vstack([low_energy_norm, rolloff_row, f0_row]).astype(np.float32)
    extra_feats = extra_feats[np.newaxis, :, :]  # (1,3,T)

    # global features
    feat_vec = compute_global_features(chunk, TARGET_SR, median_f0_hz, voiced_frac)
    global_feats = feat_vec[np.newaxis, :]  # (1,D)

    return mel, extra_feats, global_feats


# ----------------- PREDICTION -----------------


def predict_for_waveform(y: np.ndarray, model: AudioCNNv6,
                         device, idx_to_label):
    """
    Run model on all chunks of y (44.1k mono) and return:
      probs: (num_classes,)
    """
    model.eval()
    logits_list = []

    with torch.no_grad():
        for chunk in chunk_signal(y, TARGET_SR, SEGMENT_SECONDS, HOP_SECONDS):
            mel, extra, glob = chunk_to_model_inputs(chunk)
            mel_t = torch.from_numpy(mel).to(device)
            extra_t = torch.from_numpy(extra).to(device)
            glob_t = torch.from_numpy(glob).to(device)

            logits = model(mel_t, extra_t, glob_t)  # (1, C)
            logits_list.append(logits.squeeze(0))

        if not logits_list:
            # If file is somehow too tiny / empty, fake uniform prediction
            num_classes = len(idx_to_label)
            return np.ones(num_classes, dtype=np.float32) / num_classes

        all_logits = torch.stack(logits_list, dim=0)  # (num_chunks, C)
        mean_logits = all_logits.mean(dim=0)          # (C,)
        probs = F.softmax(mean_logits, dim=-1).cpu().numpy()
        return probs


def topk_from_probs(probs, idx_to_label, k=3):
    """Return top-k (label, prob) sorted descending."""
    idxs = np.argsort(-probs)[:k]
    return [(idx_to_label[int(i)], float(probs[int(i)])) for i in idxs]


# ----------------- SAMPLER-STYLE PITCH SHIFT -----------------


def sampler_style_pitch_shift(y: np.ndarray, semitones: float, sr: int) -> np.ndarray:
    """
    Sampler-style pitch shift via resampling (rate change), like a basic sampler:
      - Positive semitones: speed up playback (shorter duration, higher pitch)
      - Negative semitones: slow down (longer duration, lower pitch)

    This is equivalent to playing the same samples back at a different rate,
    then telling the system the SR is still `sr`.

    We achieve this by resampling from an effective original SR of (sr * rate)
    back to `sr`, so the time scaling is 1/rate and pitch changes by rate.

        rate = 2^(semitones / 12)

    Example:
        semitones = +12 -> rate = 2
        length_out = length_in * sr / (sr * 2) = length_in / 2

    which matches classic sampler behavior.
    """
    if semitones == 0 or y.size == 0:
        return y

    rate = 2.0 ** (semitones / 12.0)

    # Use librosa.resample with a scaled "orig_sr" to induce the time scaling.
    # Only the ratio (target_sr / orig_sr) matters internally.
    y_shift = librosa.resample(
        y,
        orig_sr=sr * rate,   # effective original SR
        target_sr=sr,        # treat output as standard TARGET_SR
    ).astype(np.float32)

    return y_shift


# ----------------- MAIN -----------------


def get_args():
    ap = argparse.ArgumentParser("Classify palette sounds with mel_v7 CNN + +/- 2 octaves.")
    ap.add_argument("--model", required=True, help="Path to best_model.pt")
    ap.add_argument("--labels", required=True, help="Path to label_mapping.json")
    ap.add_argument("--in_dir", required=True, help="Directory with .wav files to classify")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return ap.parse_args()


def main():
    args = get_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # load label mapping
    with open(args.labels, "r") as f:
        mapping = json.load(f)
    label_to_idx = mapping.get("label_to_idx", mapping)
    # build idx_to_label from mapping
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    num_classes = len(idx_to_label)
    print("Classes:", [idx_to_label[i] for i in range(num_classes)])

    # build model
    # global_dim is inferred by running on a dummy chunk once we load audio
    dummy = np.zeros(int(SEGMENT_SECONDS * TARGET_SR), dtype=np.float32)
    mel, extra, glob = chunk_to_model_inputs(dummy)
    global_dim = glob.shape[1]

    model = AudioCNNv6(
        n_classes=num_classes,
        extra_dim=extra.shape[1],
        global_dim=global_dim
    ).to(device)

    # load weights
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {args.model}")

    # collect wav files
    in_dir = Path(args.in_dir)
    wavs = sorted(in_dir.glob("*.wav"))
    print(f"Found {len(wavs)} wav files under {in_dir}")

    variants = {
        "orig": 0,     # 0 semitones
        "down2": -24,  # -24 semitones
        "up2": 24,     # +24 semitones
    }

    with open(args.out_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "file", "variant", "primary_label", "primary_conf",
                "top1", "p1", "top2", "p2", "top3", "p3"
            ]
        )

        for wav_path in wavs:
            print(f"\nProcessing {wav_path.name}")
            try:
                y = load_audio_mono(wav_path, TARGET_SR)
            except Exception as e:
                print(f"  [WARN] failed to load {wav_path}: {e}")
                continue

            for variant_name, semitones in variants.items():
                if semitones == 0:
                    y_var = y
                else:
                    try:
                        # Sampler-style pitch shift (rate change via resampling)
                        y_var = sampler_style_pitch_shift(y, semitones, TARGET_SR)
                    except Exception as e:
                        print(f"  [WARN] sampler-style pitch_shift {variant_name} failed: {e}")
                        continue

                probs = predict_for_waveform(y_var, model, device, idx_to_label)
                top3 = topk_from_probs(probs, idx_to_label, k=3)
                primary_label, primary_conf = top3[0]

                print(
                    f"  {variant_name}: {primary_label} ({primary_conf:.3f}) "
                    f"| top3 = {', '.join(f'{lab} {p:.3f}' for lab, p in top3)}"
                )

                # write row
                writer.writerow([
                    wav_path.name,
                    variant_name,
                    primary_label,
                    f"{primary_conf:.6f}",
                    top3[0][0], f"{top3[0][1]:.6f}",
                    top3[1][0], f"{top3[1][1]:.6f}",
                    top3[2][0], f"{top3[2][1]:.6f}",
                ])

    print(f"\nDone. Wrote predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
