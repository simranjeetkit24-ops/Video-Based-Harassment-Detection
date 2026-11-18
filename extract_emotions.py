"""
extract_emotions.py
For each clip: sample frames -> detect the largest face -> run the HSEmotion
model -> build clip-level features (mean, max, std per emotion) and save them
to emotion_features.csv.

Usage (PowerShell):
    python extract_emotions.py
Different dataset location:
    python extract_emotions.py --root "D:\\path\\to\\dataset"
Quick test on a few clips only:
    python extract_emotions.py --limit 20
"""
import argparse
import csv
import glob
import os
import time
import urllib.request  # hsemotion_onnx does not import this itself; the model download fails without it

import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

DEFAULT_ROOT = r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset"
MODEL_NAME = "enet_b0_8_best_vgaf"
# Expected class order of the 8-class model (not verified against the library docs).
EMOTIONS_8 = ["anger", "contempt", "disgust", "fear",
              "happiness", "neutral", "sadness", "surprise"]


def largest_face(detector, frame):
    """Return the crop of the largest detected face, or None if no face is found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(0.1 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1 = min(x + w + pad, frame.shape[1])
    y1 = min(y + h + pad, frame.shape[0])
    return frame[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--frames", type=int, default=16, help="frames sampled per clip")
    ap.add_argument("--limit", type=int, default=0, help="for testing: process only this many clips")
    ap.add_argument("--out", default="emotion_features.csv")
    args = ap.parse_args()

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    fer = HSEmotionRecognizer(model_name=MODEL_NAME)

    files = []
    for split in ("train", "test"):
        for cls, label in (("Non_Harassment", 0), ("Harassment", 1)):
            for f in sorted(glob.glob(os.path.join(args.root, split, cls, "*.mp4"))):
                files.append((f, split, cls, label))
    if not files:
        print("No clips found. Check the --root path:", args.root)
        return
    if args.limit:
        # take evenly spaced clips so both classes and both splits are represented
        files = files[:: max(len(files) // args.limit, 1)][: args.limit]
    print(f"Found {len(files)} clips. Model: {MODEL_NAME}")

    rows, names = [], None
    t0 = time.time()
    for i, (f, split, cls, label) in enumerate(files, 1):
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(n - 1, 0), args.frames).astype(int)
        probs = []
        for k in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, fr = cap.read()
            if not ok:
                continue
            face = largest_face(detector, fr)
            if face is None or face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            try:
                _, scores = fer.predict_emotions(face_rgb, logits=False)
            except Exception as e:
                print("  emotion error", os.path.basename(f), e)
                continue
            probs.append(np.asarray(scores, dtype=np.float32).ravel())
        cap.release()

        if probs:
            P = np.stack(probs)
            if names is None:
                d = P.shape[1]
                base = EMOTIONS_8 if d == 8 else [f"e{j}" for j in range(d)]
                names = ([f"{b}_mean" for b in base] +
                         [f"{b}_max" for b in base] +
                         [f"{b}_std" for b in base])
                print(f"The emotion model returned {d} scores per face.")
            feats = np.concatenate([P.mean(0), P.max(0), P.std(0)])
        else:
            feats = None

        rows.append((f, split, cls, label, len(probs), feats))
        if i % 25 == 0 or i == len(files):
            el = time.time() - t0
            print(f"{i}/{len(files)} clips | {el:.0f}s | last: {os.path.basename(f)} "
                  f"({len(probs)} faces)")

    if names is None:
        print("No face or emotion was found in any clip. The script needs to be checked.")
        return

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "split", "class", "label", "has_face",
                    "n_face_frames"] + names)
        for f, split, cls, label, nf, feats in rows:
            if feats is None:
                w.writerow([f, split, cls, label, 0, 0] + [""] * len(names))
            else:
                w.writerow([f, split, cls, label, 1, nf] + [f"{v:.5f}" for v in feats])

    no_face = sum(1 for r in rows if r[5] is None)
    print(f"\nSaved {args.out} | {len(rows)} clips | clips with no face found: {no_face}")


if __name__ == "__main__":
    main()