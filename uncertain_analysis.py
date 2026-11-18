"""
uncertain_analysis.py
Two quick checks that support the discussion section of the report.

1) Where does emotion help? Compares video, emotion and fusion accuracy on the test
   clips where TimeSformer is uncertain, and on the clips TimeSformer gets wrong.
2) Which emotion features does the emotion classifier rely on? Prints the largest
   logistic-regression coefficients (positive = pushes towards harassment).

Usage (PowerShell):
    python uncertain_analysis.py
    python uncertain_analysis.py --lo 0.2 --hi 0.8
Inputs: fusion_results.csv, emotion_features.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def acc(y, p):
    return float(((p >= 0.5).astype(int) == y).mean()) if len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default="fusion_results.csv")
    ap.add_argument("--features", default="emotion_features.csv")
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.9)
    args = ap.parse_args()

    df = pd.read_csv(args.fusion)
    y = df["label"].values
    pv, pe, pf = df["p_video"].values, df["p_emotion"].values, df["p_fusion_cv"].values

    print(f"All {len(df)} test clips: video {acc(y, pv):.3f} | emotion {acc(y, pe):.3f} "
          f"| fusion (CV) {acc(y, pf):.3f}")

    mask = (pv >= args.lo) & (pv <= args.hi)
    print(f"\n1a) Clips where TimeSformer is uncertain ({args.lo} <= P(video) <= {args.hi}): {mask.sum()}")
    if mask.sum():
        print(f"    video {acc(y[mask], pv[mask]):.3f} | emotion {acc(y[mask], pe[mask]):.3f} "
              f"| fusion {acc(y[mask], pf[mask]):.3f}")
    print("    (a small subset, so treat these numbers as exploratory only)")

    wrong = (pv >= 0.5).astype(int) != y
    print(f"\n1b) Clips TimeSformer gets wrong: {wrong.sum()}")
    if wrong.sum():
        print(f"    emotion is correct on {(((pe >= 0.5).astype(int) == y) & wrong).sum()} of them "
              f"({acc(y[wrong], pe[wrong]):.3f}); fusion is correct on "
              f"{(((pf >= 0.5).astype(int) == y) & wrong).sum()}")
        print("    If emotion is also wrong on most of these clips, the video errors are probably not face-related.")

    ef = pd.read_csv(args.features)
    cols = [c for c in ef.columns if c.endswith(("_mean", "_max", "_std"))] + ["has_face"]
    tr = ef[ef["split"] == "train"]
    X = tr[cols].fillna(tr[cols].median()).values
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X), tr["label"].values)
    order = np.argsort(-np.abs(m.coef_[0]))[:10]
    print("\n2) Strongest emotion features (standardised coefficients, train set):")
    for i in order:
        print(f"    {cols[i]:<22}{m.coef_[0][i]:+.3f}")
    print("    Negative-affect features (anger, disgust, fear, sadness) with positive weights "
          "would be consistent with harassment. A spread of unrelated features would suggest "
          "the classifier is picking up scene differences rather than emotion.")


if __name__ == "__main__":
    main()