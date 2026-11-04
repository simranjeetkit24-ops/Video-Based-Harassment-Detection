"""
Evaluate a saved fine-tuned model on the test folder.
Must be in the same folder as train_transformers.py.

    python evaluate_saved.py --model timesformer
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from train_transformers import (CLASS_NAMES, DEFAULT_ROOT, MODEL_CONFIGS,
                                SHVDataset, collect_videos, run_epoch)

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=MODEL_CONFIGS, required=True)
ap.add_argument("--root", default=DEFAULT_ROOT)
ap.add_argument("--output_dir", default="outputs")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = Path(args.output_dir) / args.model / "best_model"
cfg = MODEL_CONFIGS[args.model]

proc = AutoImageProcessor.from_pretrained(path)
model = AutoModelForVideoClassification.from_pretrained(path).to(device)

test_s = collect_videos(Path(args.root) / "test")
ds = SHVDataset(test_s, cfg["num_frames"], 224, proc.image_mean, proc.image_std, False)
loader = DataLoader(ds, batch_size=2, shuffle=False)

loss, acc, preds, gts = run_epoch(model, loader, device, nn.CrossEntropyLoss())
print(f"\nTest videos: {len(test_s)}")
print(f"Loss {loss:.4f} | Accuracy {acc*100:.2f}% | Macro F1 {f1_score(gts, preds, average='macro'):.3f}")
print("Confusion matrix (rows=true, cols=pred) [Non_Harassment, Harassment]:")
print(confusion_matrix(gts, preds))
print(classification_report(gts, preds, target_names=CLASS_NAMES, digits=3))