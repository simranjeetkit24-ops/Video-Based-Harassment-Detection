"""
Fine-tune VideoMAE / ViViT / TimeSformer on the SHV dataset using the
existing folder layout:

    DATASET_ROOT/
        train/Harassment/*.mp4
        train/Non_Harassment/*.mp4
        test/Harassment/*.mp4
        test/Non_Harassment/*.mp4

Run (from PowerShell):
    python train_transformers.py --model timesformer
    python train_transformers.py --model videomae
    python train_transformers.py --model vivit
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForVideoClassification

DEFAULT_ROOT = r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset"

MODEL_CONFIGS = {
    "videomae": {"checkpoint": "MCG-NJU/videomae-base-finetuned-kinetics", "num_frames": 16},
    "vivit": {"checkpoint": "google/vivit-b-16x2-kinetics400", "num_frames": 32},
    "timesformer": {"checkpoint": "facebook/timesformer-base-finetuned-k400", "num_frames": 8},
}

CLASSES = {"Non_Harassment": 0, "Harassment": 1}
CLASS_NAMES = ["Non_Harassment", "Harassment"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_videos(split_dir):
    samples = []
    for cls, label in CLASSES.items():
        folder = Path(split_dir) / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() == ".mp4":
                samples.append((str(p), label))
    return samples


def read_frames(path, num_frames):
    """Read the clip, sample frames uniformly, pad with last frame if short."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(frames)
    if n == 0:
        raise RuntimeError(f"Could not read video: {path}")
    if n >= num_frames:
        idx = np.linspace(0, n - 1, num_frames).astype(int)
    else:
        idx = list(range(n)) + [n - 1] * (num_frames - n)
    return np.stack([frames[i] for i in idx])


class SHVDataset(Dataset):
    def __init__(self, samples, num_frames, size, mean, std, train):
        self.samples, self.num_frames = samples, num_frames
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        if train:
            self.tf = v2.Compose([
                v2.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
                v2.RandomHorizontalFlip(0.5),
                v2.ColorJitter(0.2, 0.2, 0.2, 0.05),
                v2.RandomRotation(10),
            ])
        else:
            self.tf = v2.Resize((size, size), antialias=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        x = torch.from_numpy(read_frames(path, self.num_frames)).permute(0, 3, 1, 2).contiguous()
        x = self.tf(x).float() / 255.0
        x = (x - self.mean) / self.std
        return x, label


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, preds, gts = 0.0, [], []
    for x, y in tqdm(loader, leave=False, desc="train" if train else "eval"):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(pixel_values=x).logits
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total += loss.item() * x.size(0)
        preds += logits.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()
    return total / len(loader.dataset), accuracy_score(gts, preds), preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    ap.add_argument("--model_path", default=None, help="Local model folder (optional)")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--freeze_blocks", type=int, default=0,
                    help="Freeze embeddings + first N transformer blocks (0-12). "
                         "Higher = faster epochs, slightly lower accuracy. Try 8 on CPU.")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MODEL_CONFIGS[args.model]
    src = args.model_path or cfg["checkpoint"]
    out = Path(args.output_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Model: {args.model} | Frames: {cfg['num_frames']}")
    if device.type == "cpu":
        print("WARNING: running on CPU will be very slow. Use a GPU (e.g. Colab/Kaggle).")

    train_all = collect_videos(Path(args.root) / "train")
    test_s = collect_videos(Path(args.root) / "test")
    train_s, val_s = train_test_split(
        train_all, test_size=args.val_size,
        stratify=[l for _, l in train_all], random_state=args.seed)
    print(f"Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    proc = AutoImageProcessor.from_pretrained(src)

    def loader(s, train):
        ds = SHVDataset(s, cfg["num_frames"], 224, proc.image_mean, proc.image_std, train)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    train_loader, val_loader, test_loader = loader(train_s, True), loader(val_s, False), loader(test_s, False)

    model = AutoModelForVideoClassification.from_pretrained(
        src, num_labels=2,
        id2label=dict(enumerate(CLASS_NAMES)),
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True).to(device)

    if args.freeze_blocks > 0:
        base = model.base_model
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for blk in base.encoder.layer[:args.freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
        print(f"Frozen: embeddings + first {args.freeze_blocks} of {len(base.encoder.layer)} blocks")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler() if (args.fp16 and device.type == "cuda") else None

    history, best = [], float("inf")
    best_path = out / "best_model"
    for ep in range(1, args.epochs + 1):
        tl, ta, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        vl, va, _, _ = run_epoch(model, val_loader, device, criterion)
        scheduler.step(vl)
        history.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"Epoch {ep:02d} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")
        if vl < best:
            best = vl
            model.save_pretrained(best_path)
            proc.save_pretrained(best_path)
            print("  -> saved best model")

    model = AutoModelForVideoClassification.from_pretrained(best_path).to(device)
    tl, ta, preds, gts = run_epoch(model, test_loader, device, criterion)
    cm = confusion_matrix(gts, preds)
    mf1 = f1_score(gts, preds, average="macro")
    print("\n===== TEST RESULTS =====")
    print(f"Loss {tl:.4f} | Accuracy {ta*100:.2f}% | Macro F1 {mf1:.3f}")
    print("Confusion matrix (rows=true, cols=pred) [Non_Harassment, Harassment]:")
    print(cm)
    print(classification_report(gts, preds, target_names=CLASS_NAMES, digits=3))

    with open(out / "results.json", "w") as f:
        json.dump(dict(model=args.model, test_loss=tl, test_acc=ta, macro_f1=mf1,
                       confusion_matrix=cm.tolist(), history=history), f, indent=2)


if __name__ == "__main__":
    main()