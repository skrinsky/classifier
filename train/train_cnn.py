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
PATH_KEY = "mel_path"       # key in index.jsonl for npy path (relative to index dir)
LABEL_KEY = "class"         # key in index.jsonl for instrument/class name


# ----------------- DATASET -----------------

class MelDataset(Dataset):
    def __init__(self, records, label_to_idx, root_dir, normalize=True):
        """
        records: list of dicts loaded from index.jsonl
        label_to_idx: dict mapping label string -> integer id
        root_dir: directory that mel_path is relative to (e.g. dir of index.jsonl)
        """
        self.records = records
        self.label_to_idx = label_to_idx
        self.normalize = normalize
        self.root_dir = root_dir  # e.g. /scratch/summerk/320data/mel_data_v3

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rel_path = rec[PATH_KEY]
        label_str = rec[LABEL_KEY]

        # Resolve relative -> absolute path
        if os.path.isabs(rel_path):
            path = rel_path
        else:
            path = os.path.join(self.root_dir, rel_path)

        mel_db = np.load(path).astype(np.float32)  # (n_mels, n_frames)

        # ---- per-example normalization on the mel (in dB) ----
        mel = mel_db.copy()
        if self.normalize:
            mean = mel.mean()
            std = mel.std()
            if std < 1e-6:
                std = 1e-6
            mel = (mel - mean) / std

        # add channel dim for CNN: (1, n_mels, n_frames)
        mel = np.expand_dims(mel, axis=0)  # (1, M, T)

        # ---- extra low-frequency features (2 x T) ----
        # 1) low-band energy (first few mel bins) in approximate power domain
        # 2) spectral rolloff (85%) as a normalized bin index

        M, T = mel_db.shape
        # convert dB -> power, manually (10^(dB/10))
        power = np.power(10.0, mel_db / 10.0, dtype=np.float32)  # (M, T)

        # low-frequency band: first 8 mel bins (you can tweak)
        LOW_BINS = min(8, M)
        low_band = power[:LOW_BINS, :]                     # (LOW_BINS, T)
        low_energy = low_band.mean(axis=0, keepdims=True)  # (1, T)

        # spectral rolloff 0.85
        spec_total = power.sum(axis=0, keepdims=True) + 1e-10  # (1, T)
        cum = np.cumsum(power, axis=0)                         # (M, T)
        # for each frame, first bin where cumulative >= 0.85 * total
        thresh = 0.85 * spec_total
        rolloff_bin = (cum < thresh).sum(axis=0)               # (T,)
        rolloff_norm = rolloff_bin.astype(np.float32) / max(M - 1, 1)
        rolloff_row = rolloff_norm.reshape(1, T)               # (1, T)

        # normalize low_energy roughly to [0,1] by dividing by mean total energy
        mean_total_energy = spec_total.mean(axis=0, keepdims=True)  # (1, T)
        low_energy_norm = low_energy / (mean_total_energy + 1e-10)

        extra_feats = np.vstack([low_energy_norm, rolloff_row]).astype(np.float32)  # (2, T)

        label = self.label_to_idx[label_str]

        mel_t = torch.from_numpy(mel)          # (1, M, T)
        feats_t = torch.from_numpy(extra_feats)  # (2, T)
        label_t = torch.tensor(label, dtype=torch.long)

        # DataLoader will see this as ((mel, feats), label)
        return (mel_t, feats_t), label_t


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


class AudioCNN(nn.Module):
    def __init__(self, n_classes, extra_dim=0):
        """
        n_classes: number of instrument classes
        extra_dim: number of extra feature channels (2 for our low-freq features)
        """
        super().__init__()
        # Input mel: (B, 1, n_mels, n_frames)
        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),  # then global average pool
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.extra_dim = extra_dim
        if extra_dim > 0:
            # Conv over time axis only: (B, extra_dim, T) -> (B, 32, T)
            self.feat_conv = nn.Conv1d(extra_dim, 32, kernel_size=3, padding=1)
            fc_in = 256 + 32
        else:
            self.feat_conv = None
            fc_in = 256

        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, mel, extra_feats=None):
        """
        mel: (B, 1, n_mels, T)
        extra_feats: (B, extra_dim, T) or None
        """
        x = self.features(mel)        # (B, 256, H', W')
        x = self.gap(x)               # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 256)

        if self.extra_dim > 0 and extra_feats is not None:
            # extra_feats: (B, extra_dim, T)
            f = self.feat_conv(extra_feats)          # (B, 32, T)
            f = F.adaptive_avg_pool1d(f, 1).squeeze(-1)  # (B, 32)
            x = torch.cat([x, f], dim=1)             # (B, 256+32)

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
    # inverse-frequency weighting
    weights = np.zeros(num_classes, dtype=np.float32)
    for lab, idx in label_to_idx.items():
        freq = counts[lab] / total
        weights[idx] = 1.0 / (freq + 1e-8)
    # normalize so average weight ~ 1
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, class_weights=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for (mels, feats), labels in loader:
            mels = mels.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(mels, feats)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss_sum += loss.item() * mels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += mels.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def compute_confusion_matrix(model, loader, device, num_classes):
    """
    Returns a (num_classes, num_classes) confusion matrix:
    rows = true labels, cols = predicted labels.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for (mels, feats), labels in loader:
            mels = mels.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(mels, feats)
            preds = logits.argmax(dim=1)

            y_true = labels.cpu().numpy()
            y_pred = preds.cpu().numpy()
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
    return cm


def pretty_print_confusion_matrix(cm, idx_to_label):
    """
    Print a simple text-table version of the confusion matrix.
    """
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


# ----------------- TRAINING LOOP -----------------

def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading index from {args.index}")
    records = load_index(args.index)
    print(f"Loaded {len(records)} records")

    # build label mapping
    label_to_idx, idx_to_label = build_label_mapping(records)
    n_classes = len(label_to_idx)
    print("Classes:")
    for i, lab in sorted(idx_to_label.items()):
        print(f"  {i:2d}: {lab}")
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)

    # root_dir = directory containing index.jsonl (where mel paths are rooted)
    root_dir = os.path.dirname(os.path.abspath(args.index))

    # dataset + splits
    full_ds = MelDataset(records, label_to_idx, root_dir=root_dir, normalize=not args.no_normalize)
    train_idx, val_idx, test_idx = split_indices(len(full_ds), train_frac=0.8, val_frac=0.1, seed=args.seed)

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

    # sanity check shape
    (example_mel, example_feats), _ = full_ds[0]
    print(f"Example mel shape: {tuple(example_mel.shape)} (C, n_mels, n_frames)")
    print(f"Example extra_feats shape: {tuple(example_feats.shape)} (extra_dim, n_frames)")

    model = AudioCNN(n_classes=n_classes, extra_dim=2).to(device)

    # optional class weights
    if args.class_weights:
        class_weights = compute_class_weights(records, label_to_idx).to(device)
        print("Using class-weighted loss.")
    else:
        class_weights = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for (mels, feats), labels in train_loader:
            mels = mels.to(device)
            feats = feats.to(device)
            labels = labels.to(device)

            logits = model(mels, feats)
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

        # save best
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

    # final test eval on best model
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    else:
        print("Warning: no best_state saved; using last-epoch model for test/CM.")

    test_loss, test_acc = evaluate(model, test_loader, device, class_weights=None)
    print(f"Test  | loss={test_loss:.4f}, acc={test_acc:.4f}")

    # confusion matrix on test set
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
    ap = argparse.ArgumentParser("CNN classifier on mel_data_v* index with extra low-freq features.")
    ap.add_argument("--index", required=True, help="Path to index.jsonl (mel_data_v1/v2/v3).")
    ap.add_argument("--out_dir", required=True, help="Directory to save checkpoints and logs.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 or None disables clipping.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument("--no_normalize", action="store_true", help="Disable per-example normalization.")
    ap.add_argument("--class_weights", action="store_true", help="Use inverse-frequency class weights.")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    train(args)
