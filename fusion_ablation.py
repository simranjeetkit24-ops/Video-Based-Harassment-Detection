"""
fusion_ablation.py
Late fusion of TimeSformer (video) and facial-emotion probabilities on the test clips,
with an ablation table: video-only vs emotion-only vs video + emotion.

Why cross-validation on the test clips:
TimeSformer was fine-tuned on the train clips, so its train-set probabilities are
close to 0/1 and cannot be used to learn how to combine the two models. The test
clips are the only data where both models behave like unseen-data predictors.
The fusion model has only 3 parameters (2 weights + bias). Its accuracy is
estimated with repeated stratified 5-fold CV, so no clip is scored by a model
that was fitted on that same clip.

Usage (PowerShell):
    python fusion_ablation.py
Inputs : timesformer_probs.csv, emotion_probs.csv
Output : fusion_results.csv (per-clip probabilities)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cv_oof(X, y, repeats, seed=0):
    """Repeated 5-fold CV. Returns per-repeat accuracy/F1 and the mean out-of-fold probability."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    oof = np.zeros((repeats, len(y)))
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        r = i // 5
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[r, te] = m.predict_proba(X[te])[:, 1]
    accs = [accuracy_score(y, (oof[r] >= 0.5).astype(int)) for r in range(repeats)]
    f1s = [f1_score(y, (oof[r] >= 0.5).astype(int), average="macro") for r in range(repeats)]
    return np.array(accs), np.array(f1s), oof.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="timesformer_probs.csv")
    ap.add_argument("--emotion", default="emotion_probs.csv")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="fusion_results.csv")
    args = ap.parse_args()

    v = pd.read_csv(args.video)
    e = pd.read_csv(args.emotion)
    v = v[v["split"] == "test"][["path", "label", "p_video"]]
    e = e[e["split"] == "test"][["path", "p_emotion"]]
    df = v.merge(e, on="path", how="inner").reset_index(drop=True)
    print(f"Test clips with both probabilities: {len(df)} "
          f"(video file: {len(v)}, emotion file: {len(e)})")
    if len(df) == 0:
        print("No matching clips. The 'path' columns of the two CSVs do not match.")
        return

    y = df["label"].values
    pv, pe = df["p_video"].values, df["p_emotion"].values
    Xv = logit(pv).reshape(-1, 1)
    Xf = np.column_stack([logit(pv), logit(pe)])

    # 1) Video-only and emotion-only with the raw 0.5 threshold (no fitting on the test set)
    pred_v = (pv >= 0.5).astype(int)
    pred_e = (pe >= 0.5).astype(int)
    acc_v, f1_v = accuracy_score(y, pred_v), f1_score(y, pred_v, average="macro")
    acc_e, f1_e = accuracy_score(y, pred_e), f1_score(y, pred_e, average="macro")

    # 2) Video-only recalibrated with the SAME CV procedure as fusion (fair comparison)
    a_vr, f_vr, _ = cv_oof(Xv, y, args.repeats)
    # 3) Video + emotion fusion (CV)
    a_f, f_f, p_f = cv_oof(Xf, y, args.repeats)

    print("\nAblation table (test clips)")
    print(f"{'Setup':<42}{'Accuracy':>10}{'Macro F1':>10}")
    print(f"{'Emotion only (raw, threshold 0.5)':<42}{acc_e:>10.4f}{f1_e:>10.4f}")
    print(f"{'Video only (TimeSformer, raw 0.5)':<42}{acc_v:>10.4f}{f1_v:>10.4f}")
    print(f"{'Video only, recalibrated (5-fold CV)':<42}{a_vr.mean():>10.4f}{f_vr.mean():>10.4f}")
    print(f"{'Video + emotion fusion (5-fold CV)':<42}{a_f.mean():>10.4f}{f_f.mean():>10.4f}")
    print(f"(CV rows: mean over {args.repeats} repeats; std of accuracy: "
          f"recalibrated {a_vr.std():.4f}, fusion {a_f.std():.4f})")

    pred_f = (p_f >= 0.5).astype(int)
    print("\nFusion confusion matrix (rows = true, cols = predicted; 0 = non-harassment, 1 = harassment):")
    print(confusion_matrix(y, pred_f))

    # Paired comparison against raw video-only: which clips does fusion change?
    fixed = int(((pred_v != y) & (pred_f == y)).sum())
    broken = int(((pred_v == y) & (pred_f != y)).sum())
    print(f"\nCompared with raw video-only: fusion fixes {fixed} clips and breaks {broken} clips "
          f"(net {fixed - broken:+d} of {len(y)}).")
    print("With this few test clips, a net change of 1 to 3 clips is within noise; "
          "report it as such rather than as a proven improvement.")

    # Fitted weights on all test clips (for interpretation only, not for accuracy)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Xf, y)
    print(f"\nFitted weights on logits (all test clips): video {m.coef_[0][0]:.3f}, "
          f"emotion {m.coef_[0][1]:.3f}, bias {m.intercept_[0]:.3f}")

    df["p_fusion_cv"] = p_f
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()