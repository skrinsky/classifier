#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI to audition palette clips and classify them with mel_v7 CNN.

Features:
- Load model (.pt) and label mapping (.json)
- Auto-load default model/labels from the same folder as this script if present
- Load a WAV clip
- View waveform (orig / down2 / up2) with matplotlib
- Click in waveform to set playback start point
- Play current variant from cursor using sounddevice
- Cursor follows playback position (lightweight line update)
- When playback naturally reaches the end, cursor snaps back to 0
- Stop/Pause playback (cursor stays where it is)
- Run classification on each variant and show top-3 classes

Requires:
    pip install librosa soundfile numpy torch matplotlib sounddevice
"""

import json
import time
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import torch
import sounddevice as sd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# ---- IMPORTANT: this must match your classifier file name ----
# Assumes classify_palette_octaves2.py is in the same folder and defines:
#   TARGET_SR, AudioCNNv6, load_audio_mono, sampler_style_pitch_shift,
#   chunk_to_model_inputs, predict_for_waveform
from classify_palette.classify_palette_octaves2 import (  # type: ignore
    TARGET_SR,
    AudioCNNv6,
    load_audio_mono,
    sampler_style_pitch_shift,
    chunk_to_model_inputs,
    predict_for_waveform,
)

# Directory where this script lives
BASE_DIR = Path(__file__).resolve().parent

# Default relative paths (used for GitHub-friendly auto-loading)
DEFAULT_LABELS_PATH = BASE_DIR / "/label_mappings/label_mapping_v7.json"
DEFAULT_MODEL_PATH = BASE_DIR / "best_models/best_model_v7.pt"


class PaletteGUI:
    def __init__(self, master):
        self.master = master
        master.title("Palette Classifier GUI")

        # State
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.idx_to_label = None
        self.label_to_idx = None

        self.current_wav_path = None
        self.y_orig: np.ndarray | None = None
        self.y_down2: np.ndarray | None = None
        self.y_up2: np.ndarray | None = None

        # waveform / cursor state
        self.variant_var = tk.StringVar(value="orig")
        self.cursor_time = 0.0  # seconds

        # Matplotlib line handles for efficient updates
        self.wave_line = None
        self.cursor_line = None
        self.wave_duration = 0.0  # seconds

        # playback tracking
        self.is_playing = False
        self.play_start_time = 0.0
        self.play_start_sample = 0
        self.play_total_duration = 0.0  # seconds

        # Configure sounddevice defaults (slightly larger buffer to avoid pops)
        sd.default.samplerate = TARGET_SR
        sd.default.channels = 1
        sd.default.blocksize = 2048
        sd.default.latency = "high"

        # ---------- Top frame: model + labels ----------

        frame_model = tk.Frame(master)
        frame_model.pack(fill=tk.X, padx=8, pady=4)

        # Pre-fill with defaults (string paths)
        self.model_path_var = tk.StringVar(
            value=str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else ""
        )
        self.labels_path_var = tk.StringVar(
            value=str(DEFAULT_LABELS_PATH) if DEFAULT_LABELS_PATH.exists() else ""
        )

        tk.Label(frame_model, text="Model .pt:").grid(row=0, column=0, sticky="e")
        tk.Entry(frame_model, textvariable=self.model_path_var, width=40).grid(
            row=0, column=1, padx=4
        )
        tk.Button(frame_model, text="Browse", command=self.browse_model).grid(
            row=0, column=2, padx=4
        )

        tk.Label(frame_model, text="Labels .json:").grid(row=1, column=0, sticky="e")
        tk.Entry(frame_model, textvariable=self.labels_path_var, width=40).grid(
            row=1, column=1, padx=4
        )
        tk.Button(frame_model, text="Browse", command=self.browse_labels).grid(
            row=1, column=2, padx=4
        )

        tk.Button(
            frame_model,
            text="Load Model + Labels",
            command=self.load_model_and_labels,
        ).grid(row=2, column=0, columnspan=3, pady=4)

        # ---------- Middle frame: WAV select + variants ----------

        frame_audio = tk.Frame(master)
        frame_audio.pack(fill=tk.X, padx=8, pady=4)

        self.wav_path_var = tk.StringVar()

        tk.Label(frame_audio, text="WAV Clip:").grid(row=0, column=0, sticky="e")
        tk.Entry(frame_audio, textvariable=self.wav_path_var, width=40).grid(
            row=0, column=1, padx=4
        )
        tk.Button(frame_audio, text="Browse", command=self.browse_wav).grid(
            row=0, column=2, padx=4
        )

        tk.Button(
            frame_audio, text="Load Clip", command=self.load_clip
        ).grid(row=1, column=0, columnspan=3, pady=4)

        frame_variants = tk.Frame(master)
        frame_variants.pack(fill=tk.X, padx=8, pady=2)

        tk.Label(frame_variants, text="View / Play Variant:").grid(
            row=0, column=0, sticky="w"
        )
        tk.Radiobutton(
            frame_variants,
            text="Original",
            variable=self.variant_var,
            value="orig",
            command=lambda: self.update_waveform_full(keep_xlim=False),
        ).grid(row=0, column=1, padx=4)

        tk.Radiobutton(
            frame_variants,
            text="Down 2 (-24)",
            variable=self.variant_var,
            value="down2",
            command=lambda: self.update_waveform_full(keep_xlim=False),
        ).grid(row=0, column=2, padx=4)

        tk.Radiobutton(
            frame_variants,
            text="Up 2 (+24)",
            variable=self.variant_var,
            value="up2",
            command=lambda: self.update_waveform_full(keep_xlim=False),
        ).grid(row=0, column=3, padx=4)

        # ---------- Waveform frame (matplotlib) ----------

        frame_wave = tk.Frame(master)
        frame_wave.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_wave)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, frame_wave)
        toolbar.update()

        self.canvas.mpl_connect("button_press_event", self.on_waveform_click)

        # ---------- Playback controls ----------

        frame_play = tk.Frame(master)
        frame_play.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(
            frame_play, text="Play from Cursor", command=self.play_from_cursor
        ).grid(row=0, column=0, padx=4)

        tk.Button(
            frame_play, text="Stop / Pause", command=self.stop_playback
        ).grid(row=0, column=1, padx=4)

        # ---------- Classification ----------

        frame_classify = tk.Frame(master)
        frame_classify.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(
            frame_classify,
            text="Run Classification",
            command=self.run_classification,
        ).pack()

        # ---------- Output text box ----------

        frame_output = tk.Frame(master)
        frame_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        tk.Label(frame_output, text="Results:").pack(anchor="w")
        self.text_output = tk.Text(frame_output, height=10, width=80)
        self.text_output.pack(fill=tk.BOTH, expand=True)

        self.text_output.insert(
            tk.END,
            f"Device: {self.device}\n"
            "- By default, looks for best_model.pt and label_mapping.json\n"
            f"  in the same folder as this script: {BASE_DIR}\n"
            "- You can override them with the Browse buttons.\n"
            "Workflow:\n"
            "  1) Ensure model + labels exist, or Browse+Load them.\n"
            "  2) Load a WAV clip.\n"
            "  3) Choose variant, click in waveform to set cursor.\n"
            "  4) Play from cursor (cursor will follow) and/or run classification.\n"
            "     When playback reaches the end naturally, cursor resets to start.\n",
        )

        # Try auto-loading default model + labels on startup (silently)
        self.auto_load_defaults()

    # ---------- Playback utils ----------

    def stop_playback(self):
        sd.stop()
        self.is_playing = False
        # Cursor stays where it was when you hit stop

    # ---------- File browsing ----------

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model .pt",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path_var.set(path)

    def browse_labels(self):
        path = filedialog.askopenfilename(
            title="Select label_mapping.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.labels_path_var.set(path)

    def browse_wav(self):
        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.wav_path_var.set(path)

    # ---------- Model loading ----------

    def auto_load_defaults(self):
        """
        Try loading default model/labels on startup without popping errors.
        This makes the repo 'clone-and-run' friendly as long as the files
        are in the same folder as this script.
        """
        model_path = self.model_path_var.get().strip()
        labels_path = self.labels_path_var.get().strip()

        if not (model_path and labels_path):
            return
        if not (os.path.exists(model_path) and os.path.exists(labels_path)):
            return

        self.load_model_and_labels(silent=True)

    def load_model_and_labels(self, silent: bool = False):
        """
        Load model + labels.
        If silent=True, suppress messagebox errors (for auto-load on startup).
        """
        model_path = self.model_path_var.get().strip()
        labels_path = self.labels_path_var.get().strip()

        if not model_path or not labels_path:
            if not silent:
                messagebox.showerror("Error", "Please select both model and labels.")
            return

        try:
            with open(labels_path, "r") as f:
                mapping = json.load(f)
            label_to_idx = mapping.get("label_to_idx", mapping)
            idx_to_label = {int(v): k for k, v in label_to_idx.items()}
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Failed to load labels: {e}")
            return

        num_classes = len(idx_to_label)

        dummy = np.zeros(int(2.0 * TARGET_SR), dtype=np.float32)
        _, extra_dummy, glob_dummy = chunk_to_model_inputs(dummy)
        global_dim = glob_dummy.shape[1]
        extra_dim = extra_dummy.shape[1]

        model = AudioCNNv6(
            n_classes=num_classes,
            extra_dim=extra_dim,
            global_dim=global_dim,
        ).to(self.device)

        try:
            ckpt = torch.load(model_path, map_location=self.device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        self.model = model
        self.idx_to_label = idx_to_label
        self.label_to_idx = label_to_idx

        if not silent:
            self.text_output.insert(
                tk.END,
                f"\nLoaded model from {model_path}\n"
                f"Loaded {num_classes} classes: {list(idx_to_label.values())}\n",
            )
            self.text_output.see(tk.END)

    # ---------- Clip loading and variants ----------

    def load_clip(self):
        wav_path = self.wav_path_var.get().strip()
        if not wav_path:
            messagebox.showerror("Error", "Please select a WAV file.")
            return

        try:
            y = load_audio_mono(wav_path, TARGET_SR)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load WAV: {e}")
            return

        self.current_wav_path = wav_path
        self.y_orig = y
        self.y_down2 = sampler_style_pitch_shift(y, -24, TARGET_SR)
        self.y_up2 = sampler_style_pitch_shift(y, +24, TARGET_SR)

        self.cursor_time = 0.0
        self.update_waveform_full(keep_xlim=False)

        self.text_output.insert(
            tk.END,
            f"\nLoaded clip: {wav_path}\n"
            f"  Original length: {len(self.y_orig) / TARGET_SR:.3f} s\n"
            f"  Down2 length:    {len(self.y_down2) / TARGET_SR:.3f} s\n"
            f"  Up2 length:      {len(self.y_up2) / TARGET_SR:.3f} s\n",
        )
        self.text_output.see(tk.END)

    # ---------- Waveform plotting ----------

    def get_current_variant_waveform(self) -> np.ndarray | None:
        v = self.variant_var.get()
        if v == "orig":
            return self.y_orig
        elif v == "down2":
            return self.y_down2
        elif v == "up2":
            return self.y_up2
        return None

    def update_waveform_full(self, keep_xlim: bool = False):
        """
        Full waveform redraw (used when clip or variant changes).
        Playback cursor visual is also reset to the current cursor_time.
        """
        y = self.get_current_variant_waveform()

        old_xlim = None
        if keep_xlim and self.wave_line is not None:
            old_xlim = self.ax.get_xlim()

        self.ax.clear()
        self.ax.set_title(f"Waveform ({self.variant_var.get()})")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        self.wave_line = None
        self.cursor_line = None
        self.wave_duration = 0.0

        if y is None or y.size == 0:
            self.canvas.draw()
            return

        t = np.arange(len(y)) / float(TARGET_SR)
        self.wave_duration = float(len(y)) / float(TARGET_SR)

        # Decimate long signals for plotting
        max_points = 200000
        if len(y) > max_points:
            step = len(y) // max_points
            t_plot = t[::step]
            y_plot = y[::step]
        else:
            t_plot = t
            y_plot = y

        # Plot waveform once and keep handle
        (self.wave_line,) = self.ax.plot(t_plot, y_plot, linewidth=0.5)

        # Draw cursor line once and keep handle
        self.cursor_line = self.ax.axvline(
            self.cursor_time, color="r", linestyle="--", linewidth=1.0
        )

        if old_xlim is not None:
            self.ax.set_xlim(old_xlim)
        else:
            self.ax.set_xlim(t_plot[0], t_plot[-1])

        self.canvas.draw()

    def update_cursor_visual(self):
        """
        Lightweight cursor-only update:
        move the red line to self.cursor_time without replotting waveform.
        """
        if self.cursor_line is None:
            # No waveform plotted yet
            return

        # Move the cursor line
        self.cursor_line.set_xdata([self.cursor_time, self.cursor_time])
        self.canvas.draw_idle()

    def on_waveform_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        self.cursor_time = float(event.xdata)

        # Clamp to valid range
        if self.wave_duration > 0.0:
            if self.cursor_time < 0.0:
                self.cursor_time = 0.0
            if self.cursor_time > self.wave_duration:
                self.cursor_time = self.wave_duration

        # Only move the cursor visually (no full redraw)
        self.update_cursor_visual()

    # ---------- Playback ----------

    def play_from_cursor(self):
        y = self.get_current_variant_waveform()
        if y is None or y.size == 0:
            messagebox.showwarning("Warning", "No clip loaded for this variant.")
            return

        start_sample = int(self.cursor_time * TARGET_SR)
        if start_sample < 0:
            start_sample = 0
        if start_sample >= len(y):
            start_sample = max(0, len(y) - 1)

        segment = y[start_sample:]
        if segment.size == 0:
            messagebox.showwarning("Warning", "Cursor is at end of clip.")
            return

        sd.stop()
        sd.play(segment, TARGET_SR)

        self.is_playing = True
        self.play_start_time = time.time()
        self.play_start_sample = start_sample
        self.play_total_duration = len(y) / float(TARGET_SR)

        # Start cursor-follow with a modest update rate (100 ms)
        self.master.after(500, self.update_cursor_during_playback)

    def update_cursor_during_playback(self):
        if not self.is_playing:
            return

        elapsed = time.time() - self.play_start_time
        current_time = self.play_start_sample / float(TARGET_SR) + elapsed

        if current_time >= self.play_total_duration:
            # Natural end of playback:
            #  - stop playback
            #  - reset cursor to start (0.0)
            sd.stop()
            self.is_playing = False
            self.cursor_time = 0.0
            self.update_cursor_visual()
            return

        self.cursor_time = current_time
        self.update_cursor_visual()

        # Schedule next lightweight cursor update
        self.master.after(500, self.update_cursor_during_playback)

    # ---------- Classification ----------

    def run_classification(self):
        if self.model is None or self.idx_to_label is None:
            messagebox.showerror("Error", "Load model and labels first.")
            return
        if self.y_orig is None:
            messagebox.showerror("Error", "Load a WAV clip first.")
            return

        variants = {
            "orig": self.y_orig,
            "down2": self.y_down2,
            "up2": self.y_up2,
        }

        self.text_output.insert(
            tk.END,
            f"\n=== Classification for {self.current_wav_path} ===\n",
        )

        for name, y_var in variants.items():
            if y_var is None:
                continue
            probs = predict_for_waveform(
                y_var, self.model, self.device, self.idx_to_label
            )
            idxs = np.argsort(-probs)[:3]
            top = [
                (self.idx_to_label[int(i)], float(probs[int(i)]))
                for i in idxs
            ]

            primary_label, primary_conf = top[0]
            line = (
                f"  {name}: {primary_label} ({primary_conf:.3f}) | top3 = "
                + ", ".join(f"{lab} {p:.3f}" for lab, p in top)
                + "\n"
            )
            self.text_output.insert(tk.END, line)

        self.text_output.see(tk.END)


def main():
    root = tk.Tk()
    app = PaletteGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
