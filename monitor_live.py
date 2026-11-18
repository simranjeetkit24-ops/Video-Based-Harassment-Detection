"""
monitor_live.py
Live camera (or video file) monitor.
  * Video model (fine-tuned TimeSformer): P(harassment) over the last few seconds.
  * Facial emotion (HSEmotion): shown as context.
  * Optional --actions: top-3 generic actions (Kinetics-400 TimeSformer), shown as context.
Only the harassment probability can raise an alert.

The display never freezes: frames are shown in real time while a background thread
analyses the most recent window. On CPU one analysis takes a few seconds, so the
probability on screen is a few seconds behind the picture.

Usage (PowerShell, from the SHV_Implementation folder):
    python monitor_live.py                       # default camera (index 0)
    python monitor_live.py --source 1            # second camera
    python monitor_live.py --actions             # also show generic actions (downloads ~500 MB once)
    python monitor_live.py --source demo_long.mp4   # test on a file, played in real time
Press q in the video window to quit.
Outputs (outputs\\monitor\\): live_log.csv, alert_live_###.jpg  (only alert frames are saved).
"""
import argparse
import csv
import os
import sys
import threading
import time
import urllib.request  # noqa: F401  (hsemotion_onnx needs it imported before it downloads its model)
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer  # noqa: E402
from monitor import VideoTransformerModel, FramePreprocessor  # noqa: E402
from monitor_multimodal import draw_overlay, window_emotion, EMOTION_MODEL, MAX_W  # noqa: E402
from train_transformers import MODEL_CONFIGS  # noqa: E402

ACTION_MODEL = "facebook/timesformer-base-finetuned-k400"  # model id not verified from here


def top_actions(am, prep, frames, device, k=3):
    x = prep.process(frames)
    logits = am.model(pixel_values=x.to(device)).logits[0]
    p = logits.detach().cpu().numpy().astype(np.float64)
    p = np.exp(p - p.max())
    p /= p.sum()
    idx = np.argsort(-p)[:k]
    id2label = am.model.config.id2label
    return [(str(id2label[int(i)]), float(p[i])) for i in idx]


class Analyzer(threading.Thread):
    """Background worker: analyses the newest window whenever it is free."""

    def __init__(self, model, prep, detector, fer, am=None, act_prep=None, device="cpu"):
        super().__init__(daemon=True)
        self.model, self.prep, self.detector, self.fer = model, prep, detector, fer
        self.am, self.act_prep, self.device = am, act_prep, device
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.pending, self.result, self.busy, self.stop_flag = None, None, False, False

    def submit(self, frames, t):
        with self.lock:
            if self.busy or self.pending is not None:
                return
            self.pending = (frames, t)
            self.event.set()

    def take_result(self):
        with self.lock:
            r, self.result = self.result, None
            return r

    def run(self):
        while not self.stop_flag:
            self.event.wait(0.2)
            with self.lock:
                job, self.pending = self.pending, None
                if job is None:
                    continue
                self.event.clear()
                self.busy = True
            frames, t = job
            t0 = time.time()
            try:
                with torch.no_grad():
                    prob = float(self.model.predict(self.prep.process(frames)))
                    acts = (top_actions(self.am, self.act_prep, frames, self.device)
                            if self.am is not None else None)
                emo = window_emotion(frames, self.detector, self.fer)
                res = dict(t=t, prob=prob, emo=emo, acts=acts, sec=time.time() - t0)
            except Exception as e:  # keep the live loop alive
                res = dict(error=str(e))
            with self.lock:
                self.result, self.busy = res, False


def draw_actions(frame, acts):
    h, w = frame.shape[:2]
    k = min(max(w / 640.0, 0.5), 1.0)
    cv2.rectangle(frame, (0, 58), (w, 80), (0, 0, 0), -1)
    txt = "Actions (generic context): analysing..." if acts is None else \
        "Actions (generic context): " + ", ".join(f"{n} {p:.2f}" for n, p in acts)
    cv2.putText(frame, txt, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k,
                (180, 220, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file path")
    ap.add_argument("--model", default="timesformer")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--clip_sec", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--consecutive", type=int, default=2,
                    help="consecutive analyses >= threshold before an alert (each analysis takes seconds on CPU)")
    ap.add_argument("--actions", action="store_true", help="also show top-3 generic actions")
    ap.add_argument("--action_model", default=ACTION_MODEL)
    ap.add_argument("--no_display", action="store_true", help="no window (for tests)")
    ap.add_argument("--max_sec", type=float, default=0, help="stop after this many seconds (0 = until q)")
    ap.add_argument("--outdir", default=os.path.join("outputs", "monitor"))
    args = ap.parse_args()

    is_cam = args.source.isdigit()
    if is_cam:
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW) if os.name == "nt" \
            else cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Cannot open source:", args.source,
              "(for a camera, check that no other app uses it and that camera access is allowed in Windows settings)")
        return
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = args.model_path or Path("outputs") / args.model / "best_model"
    model = VideoTransformerModel(path, device)
    prep = FramePreprocessor(MODEL_CONFIGS[args.model]["num_frames"],
                             model.proc.image_mean, model.proc.image_std)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    fer = HSEmotionRecognizer(model_name=EMOTION_MODEL)
    am = act_prep = None
    if args.actions:
        am = VideoTransformerModel(args.action_model, device)
        act_prep = FramePreprocessor(getattr(am.model.config, "num_frames", 8),
                                     am.proc.image_mean, am.proc.image_std)

    os.makedirs(args.outdir, exist_ok=True)
    log_f = open(os.path.join(args.outdir, "live_log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["time_s", "p_video", "dominant_emotion", "p_emotion_top",
                  "negative_affect", "top_actions", "alert", "analysis_sec"])

    analyzer = Analyzer(model, prep, detector, fer, am, act_prep, device)
    analyzer.start()

    buf = deque()
    prob, emo, acts = 0.0, None, None
    streak, alert_on, n_alerts, n_an, lat_sum, idx = 0, False, 0, 0, 0.0, 0
    start = time.time()
    print(f"Source: {'camera ' + args.source if is_cam else args.source} | window {args.clip_sec}s | "
          f"alert if {args.consecutive} consecutive analyses >= {args.threshold} | device {device}"
          f"{' | actions on' if am is not None else ''}")
    print("Press q in the video window to quit." if not args.no_display else "Running without a window.")

    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                print("No more frames from the source.")
                break
            idx += 1
            t = (time.time() - start) if is_cam else idx / file_fps
            if fr.shape[1] > MAX_W:
                s = MAX_W / fr.shape[1]
                fr = cv2.resize(fr, (MAX_W, int(fr.shape[0] * s)))
            buf.append((t, fr.copy()))
            while buf and t - buf[0][0] > args.clip_sec:
                buf.popleft()
            if buf[-1][0] - buf[0][0] >= 0.9 * args.clip_sec:
                analyzer.submit([f for _, f in buf], t)

            new_alert = False
            res = analyzer.take_result()
            if res is not None:
                if "error" in res:
                    print("Analysis error:", res["error"])
                else:
                    prob, emo, acts = res["prob"], res["emo"], res["acts"]
                    n_an += 1
                    lat_sum += res["sec"]
                    streak = streak + 1 if prob >= args.threshold else 0
                    if streak >= args.consecutive and not alert_on:
                        alert_on, new_alert = True, True
                        n_alerts += 1
                    if prob < args.threshold:
                        alert_on = False
                    e_txt = "no face" if emo is None else f"{emo[0]} (neg {emo[2]:.2f})"
                    a_txt = "" if not acts else " | " + ", ".join(f"{n} {p:.2f}" for n, p in acts)
                    print(f"t={res['t']:6.1f}s  P(video)={prob:.2f}  emotion={e_txt}{a_txt}"
                          f"  [{res['sec']:.1f}s analysis]" + ("   <-- ALERT" if new_alert else ""))
                    log.writerow([f"{res['t']:.1f}", f"{prob:.4f}",
                                  "" if emo is None else emo[0],
                                  "" if emo is None else f"{emo[1]:.4f}",
                                  "" if emo is None else f"{emo[2]:.4f}",
                                  "" if not acts else "; ".join(f"{n} {p:.2f}" for n, p in acts),
                                  int(new_alert), f"{res['sec']:.2f}"])
                    log_f.flush()

            disp = draw_overlay(fr.copy(), prob, emo, alert_on)
            if am is not None:
                draw_actions(disp, acts)
            if new_alert:
                cv2.imwrite(os.path.join(args.outdir, f"alert_live_{n_alerts:03d}.jpg"), disp)

            if not args.no_display:
                cv2.imshow("Harassment monitor (press q to quit)", disp)
                wait = 1 if is_cam else max(int(1000 / file_fps), 1)
                if cv2.waitKey(wait) & 0xFF == ord("q"):
                    break
            elif not is_cam:
                time.sleep(1.0 / file_fps)   # play files in real time even without a window
            if args.max_sec and t >= args.max_sec:
                break
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop_flag = True
        cap.release()
        log_f.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    avg = lat_sum / n_an if n_an else 0.0
    print(f"\nDone. {n_an} analyses (avg {avg:.1f}s each), {n_alerts} alert(s).")
    print("Log:", os.path.join(args.outdir, "live_log.csv"))


if __name__ == "__main__":
    main()