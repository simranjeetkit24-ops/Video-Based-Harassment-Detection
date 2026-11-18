"""
timesformer_probs.py
Runs the fine-tuned TimeSformer on every clip and saves P(harassment) per clip
to timesformer_probs.csv. It reuses VideoTransformerModel and FramePreprocessor
from monitor.py, so preprocessing matches the monitoring demo.

Usage (PowerShell, from the SHV_Implementation folder):
    python timesformer_probs.py --limit 10      # quick test
    python timesformer_probs.py                 # test split only (181 clips)
    python timesformer_probs.py --splits train test
Sanity check: the TEST accuracy printed at the end should be close to the
92.27% reported for TimeSformer. If it is far lower, preprocessing is not matching.
"""
import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import VideoTransformerModel, FramePreprocessor  # noqa: E402
from train_transformers import MODEL_CONFIGS  # noqa: E402

DEFAULT_ROOT = r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset"


def read_all_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--model", default="timesformer")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="for testing: process only this many clips")
    ap.add_argument("--out", default="timesformer_probs.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = args.model_path or Path("outputs") / args.model / "best_model"
    model = VideoTransformerModel(path, device)
    prep = FramePreprocessor(MODEL_CONFIGS[args.model]["num_frames"],
                             model.proc.image_mean, model.proc.image_std)

    files = []
    for split in args.splits:
        for cls, label in (("Non_Harassment", 0), ("Harassment", 1)):
            for f in sorted(glob.glob(os.path.join(args.root, split, cls, "*.mp4"))):
                files.append((f, split, cls, label))
    if not files:
        print("No clips found. Check the --root path:", args.root)
        return
    if args.limit:
        files = files[:: max(len(files) // args.limit, 1)][: args.limit]
    print(f"Found {len(files)} clips | device: {device} | model path: {path}")

    rows = []
    t0 = time.time()
    for i, (f, split, cls, label) in enumerate(files, 1):
        frames = read_all_frames(f)
        if not frames:
            print("  could not read", os.path.basename(f))
            continue
        with torch.no_grad():
            prob = float(model.predict(prep.process(frames)))
        rows.append((f, split, label, prob))
        if i % 20 == 0 or i == len(files):
            print(f"{i}/{len(files)} clips | {time.time() - t0:.0f}s | "
                  f"last: {os.path.basename(f)} P={prob:.3f}")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "split", "label", "p_video"])
        for f, split, label, prob in rows:
            w.writerow([f, split, label, f"{prob:.6f}"])

    for split in args.splits:
        sub = [(l, p) for _, s, l, p in rows if s == split]
        if sub:
            y = np.array([l for l, _ in sub])
            p = np.array([p for _, p in sub])
            acc = ((p >= 0.5).astype(int) == y).mean()
            print(f"{split}: {len(sub)} clips | accuracy at 0.5 threshold: {acc:.4f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()