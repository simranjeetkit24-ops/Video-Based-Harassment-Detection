"""
monitor.py - prototype of the UML design (Camera -> DetectionService -> TemporalSmoother
-> AlertManager -> Dashboard -> Operator).

Slides a window over a longer video, runs the fine-tuned model on each window, raises an
alert only when several CONSECUTIVE windows are above the threshold, saves a snapshot and
clip for each alert, optionally asks the operator to confirm/dismiss, and writes an
annotated output video.

Must be in the same folder as train_transformers.py (needs outputs/<model>/best_model).

    python monitor.py --video long_clip.mp4 --model timesformer
    python monitor.py --video long_clip.mp4 --model timesformer --review
    python monitor.py --video long_clip.mp4 --threshold 0.8 --window 3 --cooldown 10

Note: the models were trained on 1-3 s clips. Behaviour on long real CCTV footage has
NOT been evaluated, so treat this as a design demo, not a validated system.
"""
import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from train_transformers import MODEL_CONFIGS


# ---------------------------------------------------------------- FramePreprocessor
class FramePreprocessor:
    def __init__(self, num_frames, mean, std, size=224):
        self.num_frames, self.size = num_frames, size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def process(self, frames):
        """frames: list of BGR images -> tensor (1, T, 3, H, W)."""
        idx = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
        arr = np.stack([cv2.cvtColor(cv2.resize(frames[i], (self.size, self.size)),
                                     cv2.COLOR_BGR2RGB) for i in idx])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        return ((x - self.mean) / self.std).unsqueeze(0)


# ---------------------------------------------------------------- VideoTransformerModel
class VideoTransformerModel:
    def __init__(self, path, device):
        self.proc = AutoImageProcessor.from_pretrained(path)
        self.model = AutoModelForVideoClassification.from_pretrained(path).to(device).eval()
        self.pos = self.model.config.label2id.get("Harassment", 1)
        self.device = device

    @torch.no_grad()
    def predict(self, x):
        logits = self.model(pixel_values=x.to(self.device)).logits
        return torch.softmax(logits, dim=-1)[0, self.pos].item()


# ---------------------------------------------------------------- TemporalSmoother
class TemporalSmoother:
    def __init__(self, window=3, threshold=0.7):
        self.window, self.threshold = window, threshold
        self.buf = deque(maxlen=window)

    def update(self, prob):
        self.buf.append(prob)

    def is_confirmed(self):
        return len(self.buf) == self.window and all(p >= self.threshold for p in self.buf)

    def reset(self):
        self.buf.clear()


# ---------------------------------------------------------------- AlertManager
class AlertManager:
    def __init__(self, out_dir, review=False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.review = review
        self.alerts = []

    def create_alert(self, t_sec, prob, frame, clip_frames, fps):
        aid = len(self.alerts) + 1
        snap = self.dir / f"alert_{aid:03d}.jpg"
        clip = self.dir / f"alert_{aid:03d}.mp4"
        cv2.imwrite(str(snap), frame)
        h, w = clip_frames[0].shape[:2]
        vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in clip_frames:
            vw.write(f)
        vw.release()

        status = "PENDING"
        print(f"\n*** ALERT #{aid} at {t_sec:.1f}s | P(harassment) = {prob:.2f} "
              f"| saved {snap.name}, {clip.name}")
        if self.review:
            ans = input("    Operator: confirm harassment? [y/n]: ").strip().lower()
            if ans.startswith("y"):
                status = "CONFIRMED"
                print("    -> CONFIRMED. (simulated) notifying Police / Response Team and logging incident.")
            else:
                status = "DISMISSED"
                print("    -> DISMISSED as false alarm. Clip kept for retraining.")
        else:
            print("    -> status PENDING (run with --review to simulate the operator step)")
        self.alerts.append(dict(alert_id=aid, time_sec=round(t_sec, 2), probability=round(prob, 4),
                                status=status, snapshot=snap.name, clip=clip.name))

    def save_log(self):
        f = self.dir / "alerts_log.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["alert_id", "time_sec", "probability",
                                               "status", "snapshot", "clip"])
            w.writeheader()
            w.writerows(self.alerts)
        return f


# ---------------------------------------------------------------- Dashboard (overlay)
COLORS = {"OK": (0, 170, 0), "WATCH": (0, 165, 255), "ALERT": (0, 0, 255)}


def draw_dashboard(frame, prob, state, t_sec):
    h, w = frame.shape[:2]
    col = COLORS[state]
    cv2.rectangle(frame, (0, 0), (w, 44), (30, 30, 30), -1)
    txt = "P(harassment): --" if prob is None else f"P(harassment): {prob:.2f}"
    cv2.putText(frame, f"{txt}   t={t_sec:5.1f}s   [{state}]", (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    bar = 0 if prob is None else int(prob * (w - 20))
    cv2.rectangle(frame, (10, 48), (w - 10, 58), (80, 80, 80), -1)
    cv2.rectangle(frame, (10, 48), (10 + bar, 58), col, -1)
    if state == "ALERT":
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: possible harassment - operator review",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", choices=["timesformer", "videomae"], default="timesformer")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--output_dir", default="outputs/monitor")
    ap.add_argument("--clip_sec", type=float, default=3.0, help="Length of each analysed window")
    ap.add_argument("--stride_sec", type=float, default=1.0, help="Step between windows")
    ap.add_argument("--window", type=int, default=3, help="Consecutive positive windows needed")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--cooldown", type=float, default=10.0, help="Seconds to suppress repeat alerts")
    ap.add_argument("--review", action="store_true", help="Ask operator to confirm/dismiss alerts")
    ap.add_argument("--no_video", action="store_true", help="Do not write the annotated video")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = args.model_path or Path("outputs") / args.model / "best_model"
    model = VideoTransformerModel(path, device)
    prep = FramePreprocessor(MODEL_CONFIGS[args.model]["num_frames"],
                             model.proc.image_mean, model.proc.image_std)
    smoother = TemporalSmoother(args.window, args.threshold)
    out_dir = Path(args.output_dir)
    manager = AlertManager(out_dir, args.review)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    win_frames = max(1, int(round(args.clip_sec * fps)))
    stride = max(1, int(round(args.stride_sec * fps)))
    print(f"Model {args.model} | {fps:.1f} fps | window {args.clip_sec}s every {args.stride_sec}s | "
          f"alert if {args.window} consecutive windows >= {args.threshold}")

    writer = None
    if not args.no_video:
        out_dir.mkdir(parents=True, exist_ok=True)
        video_out = out_dir / f"{Path(args.video).stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    buf = deque(maxlen=win_frames)
    prob, cooldown_until, evals, idx = None, -1.0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        t = idx / fps
        if len(buf) == win_frames and idx % stride == 0:
            prob = model.predict(prep.process(list(buf)))
            evals += 1
            print(f"t={t:6.1f}s  P(harassment)={prob:.2f}")
            if t >= cooldown_until:
                smoother.update(prob)
                if smoother.is_confirmed():
                    manager.create_alert(t, float(np.mean(smoother.buf)), frame, list(buf), fps)
                    smoother.reset()
                    cooldown_until = t + args.cooldown
        state = "ALERT" if t < cooldown_until else ("WATCH" if prob is not None and prob >= args.threshold else "OK")
        if writer is not None:
            writer.write(draw_dashboard(frame.copy(), prob, state, t))
        idx += 1
    cap.release()

    if evals == 0 and buf:  # video shorter than one window: analyse what we have
        prob = model.predict(prep.process(list(buf)))
        evals = 1
        print(f"Video shorter than one window. P(harassment) = {prob:.2f}")
    if writer is not None:
        writer.release()

    log = manager.save_log()
    print(f"\nDone. {evals} windows analysed, {len(manager.alerts)} alert(s). Log: {log}")
    if writer is not None:
        print(f"Annotated video: {video_out}")


if __name__ == "__main__":
    main()