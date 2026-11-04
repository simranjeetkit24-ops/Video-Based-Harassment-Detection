"""
Collect outputs/*/results.json from every trained model and produce:
    outputs/comparison_table.md / .csv
    outputs/training_curves.png
    outputs/confusion_matrices.png

    python plot_results.py
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRETTY = {
    "timesformer": "TimeSformer", "videomae": "VideoMAE", "vivit": "ViViT",
    "cnn_lstm": "CNN-LSTM", "cnn_bigru": "CNN-BiGRU", "r3d18": "3D ResNet-18",
}
ORDER = ["timesformer", "videomae", "vivit", "r3d18", "cnn_bigru", "cnn_lstm"]


def load(output_dir):
    res = {}
    for d in Path(output_dir).iterdir():
        f = d / "results.json"
        if d.is_dir() and f.exists():
            res[d.name] = json.load(open(f))
    return dict(sorted(res.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def harassment_metrics(cm):
    """cm rows=true, cols=pred, class order [Non_Harassment, Harassment]."""
    cm = np.array(cm)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()
    out = Path(args.output_dir)
    res = load(out)
    if not res:
        raise SystemExit(f"No results.json found under {out}/<model>/")

    # ---- table ----
    rows = []
    for k, r in res.items():
        p, rec, f1 = harassment_metrics(r["confusion_matrix"])
        rows.append([PRETTY.get(k, k), f"{r['test_acc']*100:.2f}", f"{r['test_loss']:.4f}",
                     f"{r['macro_f1']:.3f}", f"{p:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
    header = ["Model", "Accuracy (%)", "Test loss", "Macro F1",
              "Harassment precision", "Harassment recall", "Harassment F1"]
    with open(out / "comparison_table.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    with open(out / "comparison_table.md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    print("\n" + open(out / "comparison_table.md").read())

    # ---- training curves ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for k, r in res.items():
        h = r["history"]
        ep = [e["epoch"] for e in h]
        name = PRETTY.get(k, k)
        ax[0, 0].plot(ep, [e["train_loss"] for e in h], marker="o", ms=3, label=name)
        ax[0, 1].plot(ep, [e["val_loss"] for e in h], marker="o", ms=3, label=name)
        ax[1, 0].plot(ep, [e["train_acc"] * 100 for e in h], marker="o", ms=3, label=name)
        ax[1, 1].plot(ep, [e["val_acc"] * 100 for e in h], marker="o", ms=3, label=name)
    for a, t, y in zip(ax.flat, ["Train loss", "Validation loss", "Train accuracy", "Validation accuracy"],
                       ["Loss", "Loss", "Accuracy (%)", "Accuracy (%)"]):
        a.set_title(t); a.set_xlabel("Epoch"); a.set_ylabel(y); a.grid(alpha=0.3); a.legend()
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=150)
    plt.close()

    # ---- confusion matrices ----
    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    labels = ["Non-Harass.", "Harass."]
    for a, (k, r) in zip(axes[0], res.items()):
        cm = np.array(r["confusion_matrix"])
        a.imshow(cm, cmap="Blues")
        a.set_title(f"{PRETTY.get(k, k)}\nacc {r['test_acc']*100:.2f}%")
        a.set_xticks([0, 1]); a.set_xticklabels(labels)
        a.set_yticks([0, 1]); a.set_yticklabels(labels)
        a.set_xlabel("Predicted"); a.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"Saved: {out/'comparison_table.md'}, {out/'training_curves.png'}, {out/'confusion_matrices.png'}")


if __name__ == "__main__":
    main()