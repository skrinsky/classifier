#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ---- CONFIG FOR INDEX FIELDS ----
MEL_KEY   = "mel_path"     # relative path to mel npy
F0_KEY    = "f0_path"      # relative path to f0 contour npy
FEAT_KEY  = "feat_path"    # relative path to global feature npy (MFCC+stats+chroma+F0 stats)
LABEL_KEY = "class"        # instrument / class name

# ---- SIMPLE "SILENCE" FILTER CONFIG ----
# We drop segments with very low overall mel energy, for ALL classes.
# This is based on RMS over the mel *power* (10**(mel_db/10)).
MIN_MEL_RMS_DB = -55.0  # you can tweak this to -60, -50, etc.

# ==== NEW: LABEL CLEANUP CONFIG (bass vs guitar using median_f0_hz) ====
# Heuristics to reduce overlap between bass and guitar:
# - keep only clearly bass-range segments for "bass"
# - drop ultra-low "guitar" segments that are functionally bass
BASS_F0_MIN_HZ   = 25.0   # below this is weird / sub-rumble for labeled "bass"
BASS_F0_MAX_HZ   = 250.0  # above this is probably mislabeled for "bass"
GUITAR_F0_MIN_HZ = 65.0   # below this, "guitar" behaves like bass; we drop it


# ----------------- DATASET -----------------

class MelV6Dataset(Dataset):
    def __init__(self, records, label_to_idx, root_dir, normalize=True):
        """
        records: list of dicts loaded from index.jsonl
        label_to_idx: dict mapping label string -> integer id
        root_dir: directory that mel_path / f0_path / feat_path are relative to
        """
        self.records = records
        self.label_to_idx = label_to_idx
        self.normalize = normalize
        self.root_dir = root_dir

    def __len__(self):
        return len(self.records)

    def _resolve_path(self, rel_path):
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.root_dir, rel_path)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label_str = rec[LABEL_KEY]

        # ----- MEL -----
        rel_mel = rec[MEL_KEY]
        mel_path = self._resolve_path(rel_mel)
        mel_db = np.load(mel_path).astype(np.float32)  # (M_mel, T)
        M_mel, T = mel_db.shape

        mel = mel_db.copy()
        if self.normalize:
            mean = mel.mean()
            std = mel.std()
            if std < 1e-6:
                std = 1e-6
            mel = (mel - mean) / std
        mel = np.expand_dims(mel, axis=0)  # (1, M_mel, T)

        # ----- F0 contour (normalized [0,1], 0 = unvoiced) -----
        rel_f0 = rec.get(F0_KEY, None)
        if rel_f0 is not None:
            f0_path = self._resolve_path(rel_f0)
            try:
                f0 = np.load(f0_path).astype(np.float32)  # (T_f0,) or (1, T_f0)
            except Exception:
                f0 = np.zeros((T,), dtype=np.float32)
        else:
            f0 = np.zeros((T,), dtype=np.float32)

        if f0.ndim == 2:
            f0 = f0.squeeze(0)
        elif f0.ndim != 1:
            f0 = f0.reshape(-1)

        T_f0 = f0.shape[0]
        if T_f0 > T:
            f0 = f0[:T]
        elif T_f0 < T:
            pad = np.zeros((T - T_f0,), dtype=np.float32)
            f0 = np.concatenate([f0, pad], axis=0)
        f0_row = f0.reshape(1, T)  # (1, T)

        # ----- low-frequency time-varying features from mel -----
        # 1) low-band energy (first few mel bins) in power domain
        # 2) spectral rolloff (0.85) as normalized bin index
        power = np.power(10.0, mel_db / 10.0, dtype=np.float32)  # (M_mel, T)

        LOW_BINS = min(8, M_mel)
        low_band = power[:LOW_BINS, :]                     # (LOW_BINS, T)
        low_energy = low_band.mean(axis=0, keepdims=True)  # (1, T)

        spec_total = power.sum(axis=0, keepdims=True) + 1e-10  # (1, T)
        cum = np.cumsum(power, axis=0)                         # (M_mel, T)
        thresh = 0.85 * spec_total
        rolloff_bin = (cum < thresh).sum(axis=0)               # (T,)
        rolloff_norm = rolloff_bin.astype(np.float32) / max(M_mel - 1, 1)
        rolloff_row = rolloff_norm.reshape(1, T)               # (1, T)

        mean_total_energy = spec_total.mean(axis=0, keepdims=True)  # (1, T)
        low_energy_norm = low_energy / (mean_total_energy + 1e-10)  # (1, T)

        # stack time-varying extras: [low_energy_norm, rolloff, f0]
        extra_feats = np.vstack(
            [low_energy_norm, rolloff_row, f0_row]
        ).astype(np.float32)                                   # (3, T)

        # ----- global feature vector (MFCC+spectral+chroma+F0 stats) -----
        rel_feat = rec.get(FEAT_KEY, None)
        if rel_feat is not None:
            feat_path = self._resolve_path(rel_feat)
            try:
                feat_vec = np.load(feat_path).astype(np.float32)
            except Exception:
                feat_vec = np.zeros((1,), dtype=np.float32)
        else:
            feat_vec = np.zeros((1,), dtype=np.float32)

        feat_vec = feat_vec.reshape(-1).astype(np.float32)     # (D,)

        label = self.label_to_idx[label_str]

        mel_t    = torch.from_numpy(mel)         # (1, M_mel, T)
        extra_t  = torch.from_numpy(extra_feats) # (3, T)
        global_t = torch.from_numpy(feat_vec)    # (D,)
        label_t  = torch.tensor(label, dtype=torch.long)

        # DataLoader will see this as ((mel, extra, global), label)
        return (mel_t, extra_t, global_t), label_t


# ----------------- MODEL -----------------

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
        """
        n_classes: number of instrument classes
        extra_dim: number of time-varying extra channels (3: low_energy, rolloff, f0)
        global_dim: dimensionality of the global feature vector
        """
        super().__init__()

        # 2D CNN over mel
        self.mel_branch = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),  # then global avg pool
        )
        self.gap2d = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 256, 1, 1)

        # 1D conv over time-varying extras
        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.feat_conv = nn.Conv1d(extra_dim, 32, kernel_size=3, padding=1)
            self.gap1d = nn.AdaptiveAvgPool1d(1)   # -> (B, 32, 1)
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
        mel:          (B, 1, M_mel, T)
        extra_feats:  (B, extra_dim, T) or None
        global_feats: (B, global_dim) or None
        """
        # mel branch
        xm = self.mel_branch(mel)        # (B, 256, H', W')
        xm = self.gap2d(xm)              # (B, 256, 1, 1)
        xm = xm.view(xm.size(0), -1)     # (B, 256)

        pieces = [xm]

        # extras branch
        if self.extra_dim > 0 and extra_feats is not None:
            f = self.feat_conv(extra_feats)        # (B, 32, T)
            f = self.gap1d(f).squeeze(-1)          # (B, 32)
            pieces.append(f)

        # global features
        if self.global_dim > 0 and global_feats is not None:
            pieces.append(global_feats)

        x = torch.cat(pieces, dim=1)
        logits = self.fc(x)
        return logits


# ----------------- UTILS -----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_index(index_path):
    records = []
    with open(index_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    return records


def build_label_mapping(records):
    labels = sorted({rec[LABEL_KEY] for rec in records})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


def split_indices(n, train_frac=0.8, val_frac=0.1, seed=42):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def compute_class_weights(records, label_to_idx):
    counts = Counter(rec[LABEL_KEY] for rec in records)
    total = sum(counts.values())
    num_classes = len(label_to_idx)
    weights = np.zeros(num_classes, dtype=np.float32)
    for lab, idx in label_to_idx.items():
        freq = counts[lab] / total
        weights[idx] = 1.0 / (freq + 1e-8)
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, class_weights=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for (mels, extras, globals_), labels in loader:
            mels = mels.to(device)
            extras = extras.to(device)
            globals_ = globals_.to(device)
            labels = labels.to(device)

            logits = model(mels, extras, globals_)
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            loss_sum += loss.item() * mels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += mels.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def compute_confusion_matrix(model, loader, device, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for (mels, extras, globals_), labels in loader:
            mels = mels.to(device)
            extras = extras.to(device)
            globals_ = globals_.to(device)
            labels = labels.to(device)

            logits = model(mels, extras, globals_)
            preds = logits.argmax(dim=1)

            y_true = labels.cpu().numpy()
            y_pred = preds.cpu().numpy()
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
    return cm


def pretty_print_confusion_matrix(cm, idx_to_label):
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    col_width = max(len(l) for l in labels) + 2

    header = " " * (col_width) + "".join(l.ljust(col_width) for l in labels)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(header)
    for i, lab in enumerate(labels):
        row_str = lab.ljust(col_width)
        for j in range(len(labels)):
            row_str += str(cm[i, j]).ljust(col_width)
        print(row_str)
    print()


# ------------- SILENCE FILTER HELPERS -------------

def estimate_mel_rms_db(rec, root_dir):
    """
    Compute overall RMS (in dB) of the mel spectrogram for this record.
    Uses mel power: power = 10^(mel_db/10), then RMS over all bins/frames.
    """
    rel_mel = rec[MEL_KEY]
    if os.path.isabs(rel_mel):
        mel_path = rel_mel
    else:
        mel_path = os.path.join(root_dir, rel_mel)

    try:
        mel_db = np.load(mel_path).astype(np.float32)
    except Exception:
        # If we can't load it, treat as "not silent" so we don't
        # accidentally drop everything due to a read error.
        return 0.0

    power = np.power(10.0, mel_db / 10.0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(power) + 1e-12))
    rms_db = 10.0 * np.log10(rms + 1e-12)
    return rms_db


def filter_silent_records(records, root_dir, min_db):
    """
    Drop records whose mel RMS falls below min_db (for all classes).
    """
    original_n = len(records)
    dropped_counts = Counter()
    kept = []

    for rec in records:
        cls = rec.get(LABEL_KEY, "")
        rms_db = estimate_mel_rms_db(rec, root_dir)
        if rms_db < min_db:
            dropped_counts[cls] += 1
            continue
        kept.append(rec)

    print(
        f"After mel-RMS filter (min={min_db} dB): "
        f"{len(kept)} records (dropped {original_n - len(kept)} total)"
    )
    if dropped_counts:
        print("Dropped per class (mel-RMS):")
        for cls in sorted(dropped_counts.keys()):
            print(f"  {cls:12s}: {dropped_counts[cls]}")

    return kept


# ==== NEW: LABEL AMBIGUITY CLEANUP (bass vs guitar) ====

def clean_bass_guitar_by_f0(records):
    """
    Use median_f0_hz to make bass vs guitar less ambiguous.

    - For 'bass': keep only segments whose median_f0_hz is in [BASS_F0_MIN_HZ, BASS_F0_MAX_HZ]
      and not zero.
    - For 'guitar': drop segments with 0 < median_f0_hz < GUITAR_F0_MIN_HZ (they behave like bass).

    Other classes are untouched.
    """
    kept = []
    dropped_bass = 0
    dropped_gtr = 0

    for rec in records:
        cls = rec.get(LABEL_KEY, "")
        f0 = float(rec.get("median_f0_hz", 0.0))

        if cls == "bass":
            if f0 == 0.0 or f0 < BASS_F0_MIN_HZ or f0 > BASS_F0_MAX_HZ:
                dropped_bass += 1
                continue
        elif cls == "guitar":
            if 0.0 < f0 < GUITAR_F0_MIN_HZ:
                dropped_gtr += 1
                continue

        kept.append(rec)

    print(
        "After bass/guitar median_f0 cleanup: "
        f"{len(kept)} records "
        f"(dropped {dropped_bass} bass, {dropped_gtr} guitar)"
    )
    return kept


# ----------------- TRAINING LOOP -----------------

def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading index from {args.index}")
    records = load_index(args.index)
    print(f"Loaded {len(records)} records")

    # root dir for resolving npy paths
    root_dir = os.path.dirname(os.path.abspath(args.index))

    # ---- FILTER 1: drop near-silent segments by mel RMS, for ALL classes ----
    records = filter_silent_records(records, root_dir, MIN_MEL_RMS_DB)

    # ---- FILTER 2 (NEW): bass/guitar ambiguity cleanup using median_f0_hz ----
    records = clean_bass_guitar_by_f0(records)

    if not records:
        raise RuntimeError("No records left after filtering; check thresholds.")

    # label mapping (after filtering)
    label_to_idx, idx_to_label = build_label_mapping(records)
    n_classes = len(label_to_idx)
    print("Classes:")
    for i, lab in sorted(idx_to_label.items()):
        print(f"  {i:2d}: {lab}")
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w") as f:
        json.dump(
            {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label},
            f,
            indent=2,
        )

    # figure out global feature dim from first record
    first_feat_rel = records[0].get(FEAT_KEY)
    if first_feat_rel is None:
        global_dim = 0
    else:
        first_feat_path = first_feat_rel if os.path.isabs(first_feat_rel) \
            else os.path.join(root_dir, first_feat_rel)
        feat_vec0 = np.load(first_feat_path).astype(np.float32).reshape(-1)
        global_dim = int(feat_vec0.shape[0])
    print(f"Global feature dim: {global_dim}")

    # dataset + splits
    full_ds = MelV6Dataset(
        records,
        label_to_idx,
        root_dir=root_dir,
        normalize=not args.no_normalize,
    )
    train_idx, val_idx, test_idx = split_indices(
        len(full_ds), train_frac=0.8, val_frac=0.1, seed=args.seed
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # sanity check shapes
    (example_mel, example_extra, example_global), _ = full_ds[0]
    print(f"Example mel shape:    {tuple(example_mel.shape)}    (C, n_mels, n_frames)")
    print(f"Example extra shape:  {tuple(example_extra.shape)}  (extra_dim, n_frames)")
    print(f"Example global shape: {tuple(example_global.shape)} (global_dim,)")

    model = AudioCNNv6(
        n_classes=n_classes,
        extra_dim=example_extra.shape[0],
        global_dim=global_dim,
    ).to(device)

    if args.class_weights:
        class_weights = compute_class_weights(records, label_to_idx).to(device)
        print("Using class-weighted loss.")
    else:
        class_weights = None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for (mels, extras, globals_), labels in train_loader:
            mels = mels.to(device)
            extras = extras.to(device)
            globals_ = globals_.to(device)
            labels = labels.to(device)

            logits = model(mels, extras, globals_)
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * mels.size(0)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += mels.size(0)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, class_weights=None)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            out_path = os.path.join(args.out_dir, "best_model.pt")
            print(f"[SAVE] New best model at epoch {epoch:03d} (val_acc={val_acc:.4f}) -> {out_path}")
            torch.save(best_state, out_path)

    print(f"Best val_acc: {best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    else:
        print("Warning: no best_state saved; using last-epoch model for test/CM.")

    test_loss, test_acc = evaluate(model, test_loader, device, class_weights=None)
    print(f"Test  | loss={test_loss:.4f}, acc={test_acc:.4f}")

    cm = compute_confusion_matrix(model, test_loader, device, n_classes)
    cm_path = os.path.join(args.out_dir, "confusion_matrix.npy")
    np.save(cm_path, cm)
    print(f"[SAVE] Confusion matrix saved to {cm_path}")
    pretty_print_confusion_matrix(cm, idx_to_label)

    metrics_path = os.path.join(args.out_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
            f,
            indent=2,
        )
    print(f"[SAVE] Final metrics saved to {metrics_path}")


# ----------------- MAIN / ARGPARSE -----------------

def get_args():
    ap = argparse.ArgumentParser(
        "CNN on mel_data_v6_dual index with mel + time-varying extras + global MFCC/spectral/chroma/F0 features."
    )
    ap.add_argument("--index", required=True, help="Path to index.jsonl (e.g., mel_data_v6_dual/index.jsonl).")
    ap.add_argument("--out_dir", required=True, help="Directory to save checkpoints and logs.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)  # default: no weight decay
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 or None disables clipping.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument("--no_normalize", action="store_true", help="Disable per-example mel normalization.")
    ap.add_argument("--class_weights", action="store_true", help="Use inverse-frequency class weights.")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    train(args)
