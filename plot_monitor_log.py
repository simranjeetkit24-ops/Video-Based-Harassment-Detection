"""
plot_monitor_log.py
Plots P(video) over time with the alert threshold, the alert moment, and the facial
negative-affect score (anger + disgust + fear + sadness) from multimodal_log.csv.

Usage (PowerShell):
    python plot_monitor_log.py
    python plot_monitor_log.py --gt_start 7.7 --gt_end 22.1    # optional: shade the known harassment part
Output: outputs\\monitor\\multimodal_plot.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join("outputs", "monitor", "multimodal_log.csv"))
    ap.add_argument("--out", default=os.path.join("outputs", "monitor", "multimodal_plot.png"))
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--gt_start", type=float, default=None, help="seconds where harassment clips start")
    ap.add_argument("--gt_end", type=float, default=None, help="seconds where harassment clips end")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    t = df["time_s"].values

    fig, ax = plt.subplots(figsize=(9, 4.2))
    if args.gt_start is not None and args.gt_end is not None:
        ax.axvspan(args.gt_start, args.gt_end, color="orange", alpha=0.15,
                   label="Harassment clips in demo video")
    ax.plot(t, df["p_video"], color="tab:red", marker="o", ms=3.5, label="P(harassment), video")
    ax.axhline(args.threshold, color="gray", ls="--", lw=1, label=f"Alert threshold ({args.threshold})")
    for ta in df.loc[df["alert"] == 1, "time_s"]:
        ax.axvline(ta, color="black", lw=1.5, label="Alert raised")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(harassment)")
    ax.set_ylim(-0.02, 1.05)

    ax2 = ax.twinx()
    ax2.plot(t, df["negative_affect"], color="tab:blue", marker="s", ms=3, lw=1,
             alpha=0.7, label="Facial negative affect")
    ax2.set_ylabel("Negative affect (anger+disgust+fear+sadness)")
    ax2.set_ylim(-0.02, 1.05)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    ax.legend(H, L, loc="upper left", fontsize=8)
    ax.set_title("Video probability and facial negative affect over time")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("Saved", args.out)


if __name__ == "__main__":
    main()