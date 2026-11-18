"""
Joins a few SHV test clips into one longer video (demo_long.mp4) for monitor.py.

    python make_demo_video.py
    python make_demo_video.py --root "D:\\path\\to\\Sexual_Harassment_Dataset\\test"
"""
import argparse
import glob
import os

import cv2

DEFAULT_ROOT = r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset\test"

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=DEFAULT_ROOT, help="Path to the dataset's test folder")
ap.add_argument("--out", default="demo_long.mp4")
args = ap.parse_args()

normal = sorted(glob.glob(os.path.join(args.root, "Non_Harassment", "*.mp4")))
harass = sorted(glob.glob(os.path.join(args.root, "Harassment", "*.mp4")))
print(f"Found {len(normal)} non-harassment and {len(harass)} harassment clips")
if len(normal) < 3 or len(harass) < 4:
    raise SystemExit("Not enough clips found. Check --root (it must contain Harassment and Non_Harassment).")

# 2 normal, 4 harassment, 1 normal
clips = normal[:2] + harass[:4] + normal[2:3]

out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
for c in clips:
    cap = cv2.VideoCapture(c)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.resize(frame, (640, 360)))
        n += 1
    cap.release()
    print(f"added {os.path.basename(c)} ({n} frames)")
out.release()
print(f"Saved {args.out}")