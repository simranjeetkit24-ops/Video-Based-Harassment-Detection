"""
train_emotion_classifier.py
Trains an emotion-only harassment classifier on emotion_features.csv.

- Uses the existing train/test split from the dataset folders.
- Missing features (clips with no face found) are filled with the train-set
  median, and a has_face column is kept as an extra feature.
- Train-set probabilities are out-of-fold (5-fold cross-validation), so they can
  later be used to fit the fusion step without the leakage a plain train-set
  prediction would have. Test probabilities come from a model fit on all train data.

Usage (PowerShell):
    python -m pip install scikit-learn pandas
    python train_emotion_classifier.py
    python train_emotion_classifier.py --model rf
Output: emotion_probs.csv (path, split, label, p_emotion)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(kind):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                      random_state=0, n_jobs=-1)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="emotion_features.csv")
    ap.add_argument("--model", choices=["lr", "rf"], default="lr")
    ap.add_argument("--out", default="emotion_probs.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feat_cols = [c for c in df.columns
                 if c.endswith(("_mean", "_max", "_std"))] + ["has_face"]

    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Train: {len(tr)} clips | Test: {len(te)} clips")

    med = tr[feat_cols].median()
    Xtr = tr[feat_cols].fillna(med).values
    Xte = te[feat_cols].fillna(med).values
    ytr, yte = tr["label"].values, te["label"].values

    model = build_model(args.model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    p_tr = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    print(f"5-fold CV accuracy on train: {accuracy_score(ytr, p_tr >= 0.5):.3f}")

    model.fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    pred = (p_te >= 0.5).astype(int)

    print(f"\nTEST accuracy: {accuracy_score(yte, pred):.4f} | "
          f"macro F1: {f1_score(yte, pred, average='macro'):.4f}")
    print("Confusion matrix (rows = true, cols = predicted; 0 = non-harassment, 1 = harassment):")
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred,
                                target_names=["Non_Harassment", "Harassment"],
                                digits=3))

    out = pd.concat([
        tr.assign(p_emotion=p_tr)[["path", "split", "label", "p_emotion"]],
        te.assign(p_emotion=p_te)[["path", "split", "label", "p_emotion"]],
    ])
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()