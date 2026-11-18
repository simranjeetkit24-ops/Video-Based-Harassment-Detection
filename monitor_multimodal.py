"""
monitor_multimodal.py
Sliding-window harassment monitor that combines two sources of information:
  * Video (fine-tuned TimeSformer) decides the alert.
  * Facial emotion (HSEmotion) is shown as supporting context for the operator.
Emotion does not trigger or suppress alerts, because the ablation showed that fusing
it did not improve accuracy over video alone.

Reuses VideoTransformerModel and FramePreprocessor from monitor.py.

Usage (PowerShell, from the SHV_Implementation folder):
    python monitor_multimodal.py --video demo_long.mp4
Options: --clip_sec 3 --stride_sec 1 --threshold 0.7 --consecutive 3
Outputs (in outputs\\monitor\\): <name>_multimodal.mp4, multimodal_log.csv, alert_mm_###.jpg
"""
import argparse
import csv
import os
import sys
import urllib.request  # hsemotion_onnx does not import this itself; its model download fails without it
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer  # noqa: E402
from monitor import VideoTransformerModel, FramePreprocessor  # noqa: E402
from train_transformers import MODEL_CONFIGS  # noqa: E402

EMOTION_MODEL = "enet_b0_8_best_vgaf"
EMOTIONS_8 = ["anger", "contempt", "disgust", "fear",
              "happiness", "neutral", "sadness", "surprise"]  # order not verified against library docs
NEGATIVE = ["anger", "disgust", "fear", "sadness"]
MAX_W = 640


def largest_face(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(0.1 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
    return frame[y0:y1, x0:x1]


def window_emotion(frames, detector, fer, n=4):
    """Average emotion probabilities over n frames of the window. Returns (label, p_top, negative_score) or None."""
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)
    probs = []
    for i in idxs:
        face = largest_face(detector, frames[i])
        if face is None or face.size == 0:
            continue
        try:
            _, s = fer.predict_emotions(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), logits=False)
            probs.append(np.asarray(s, dtype=np.float32).ravel())
        except Exception:
            continue
    if not probs:
        return None
    m = np.mean(probs, axis=0)
    if len(m) == 8:
        names = EMOTIONS_8
        neg = float(sum(m[names.index(k)] for k in NEGATIVE))
    else:
        names = [f"e{j}" for j in range(len(m))]
        neg = float("nan")
    k = int(np.argmax(m))
    return names[k], float(m[k]), neg


def draw_overlay(frame, prob, emo, alert):
    h, w = frame.shape[:2]
    k = min(max(w / 640.0, 0.5), 1.0)  # shrink text on narrow videos
    cv2.rectangle(frame, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.putText(frame, f"Video P(harassment): {prob:.2f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (255, 255, 255), 1, cv2.LINE_AA)
    bar_w = int(min(max(prob, 0.0), 1.0) * (w - 20))
    cv2.rectangle(frame, (10, 30), (w - 10, 42), (80, 80, 80), 1)
    cv2.rectangle(frame, (10, 30), (10 + bar_w, 42), (0, 0, 255) if prob >= 0.7 else (0, 200, 0), -1)
    txt = "Emotion (context): no face" if emo is None else \
        f"Emotion (context): {emo[0]} {emo[1]:.2f} | negative {emo[2]:.2f}"
    cv2.putText(frame, txt, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k,
                (200, 200, 200), 1, cv2.LINE_AA)
    if alert:
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, "ALERT: possible harassment", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="timesformer")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--clip_sec", type=float, default=3.0)
    ap.add_argument("--stride_sec", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.join("outputs", "monitor"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open video:", args.video)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_W / W0)
    W, H = int(W0 * scale), int(H0 * scale)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = args.model_path or Path("outputs") / args.model / "best_model"
    model = VideoTransformerModel(path, device)
    prep = FramePreprocessor(MODEL_CONFIGS[args.model]["num_frames"],
                             model.proc.image_mean, model.proc.image_std)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    fer = HSEmotionRecognizer(model_name=EMOTION_MODEL)

    os.makedirs(args.outdir, exist_ok=True)
    name = Path(args.video).stem
    out_path = os.path.join(args.outdir, f"{name}_multimodal.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    win = int(round(args.clip_sec * fps))
    stride = max(int(round(args.stride_sec * fps)), 1)
    buf = deque(maxlen=win)
    prob, emo, streak, alert_on, n_alerts, n_win = 0.0, None, 0, False, 0, 0
    log = []
    print(f"Video: {fps:.1f} fps | window {args.clip_sec}s every {args.stride_sec}s | "
          f"alert if {args.consecutive} consecutive windows >= {args.threshold} | device {device}")

    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if scale < 1.0:
            fr = cv2.resize(fr, (W, H))
        buf.append(fr.copy())
        idx += 1
        if len(buf) == win and (idx - win) % stride == 0:
            frames = list(buf)
            with torch.no_grad():
                prob = float(model.predict(prep.process(frames)))
            emo = window_emotion(frames, detector, fer)
            t = idx / fps
            n_win += 1
            streak = streak + 1 if prob >= args.threshold else 0
            new_alert = False
            if streak >= args.consecutive and not alert_on:
                alert_on, new_alert = True, True
                n_alerts += 1
            if prob < args.threshold:
                alert_on = False
            e_txt = "no face" if emo is None else f"{emo[0]} ({emo[1]:.2f}, neg {emo[2]:.2f})"
            print(f"t={t:6.1f}s  P(video)={prob:.2f}  emotion={e_txt}"
                  + ("   <-- ALERT" if new_alert else ""))
            log.append([f"{t:.1f}", f"{prob:.4f}",
                        "" if emo is None else emo[0],
                        "" if emo is None else f"{emo[1]:.4f}",
                        "" if emo is None else f"{emo[2]:.4f}",
                        int(new_alert)])
            if new_alert:
                snap = draw_overlay(fr.copy(), prob, emo, True)
                cv2.imwrite(os.path.join(args.outdir, f"alert_mm_{n_alerts:03d}.jpg"), snap)
        writer.write(draw_overlay(fr.copy(), prob, emo, alert_on))

    cap.release()
    writer.release()
    with open(os.path.join(args.outdir, "multimodal_log.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "p_video", "dominant_emotion", "p_emotion_top",
                    "negative_affect", "alert"])
        w.writerows(log)
    print(f"\nDone. {n_win} windows analysed, {n_alerts} alert(s).")
    print(f"Annotated video: {out_path}")
    print(f"Log: {os.path.join(args.outdir, 'multimodal_log.csv')}")


if __name__ == "__main__":
    main()