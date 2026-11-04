"""
Baseline models surveyed in the review paper (Dhuri & Patil, 2025), trained on
the same SHV data/splits as the transformers so results are directly comparable.

    cnn_lstm    ResNet-18 features -> LSTM          (cf. SHV paper baseline [1], Pallewar et al. [31])
    cnn_bigru   ResNet-18 features -> BiGRU         (cf. Gandapur et al. [30], Qasim et al. [44])
    r3d18       3D ResNet-18, Kinetics-400 weights  (cf. Srividya et al. [32], Al-Madani et al. [33])

Must be in the same folder as train_transformers.py.

    python baselines.py --model cnn_lstm
    python baselines.py --model cnn_bigru
    python baselines.py --model r3d18

Note: torchvision downloads the pretrained weights on first run (needs internet).
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models

from train_transformers import (CLASS_NAMES, DEFAULT_ROOT, SHVDataset,
                                collect_videos, run_epoch, set_seed)

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
KINETICS = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])


class ResNetEncoder(nn.Module):
    """Per-frame ResNet-18 feature extractor: (B,T,C,H,W) -> (B,T,512)."""

    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        b, t, c, h, w = x.shape
        f = self.net(x.reshape(b * t, c, h, w))
        return f.view(b, t, -1)


class CNNLSTM(nn.Module):
    def __init__(self, hidden=256, dropout=0.3):
        super().__init__()
        self.enc = ResNetEncoder()
        self.rnn = nn.LSTM(512, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        out, _ = self.rnn(self.enc(x))
        return self.head(out[:, -1])


class CNNBiGRU(nn.Module):
    def __init__(self, hidden=256, dropout=0.3):
        super().__init__()
        self.enc = ResNetEncoder()
        self.rnn = nn.GRU(512, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 2))

    def forward(self, x):
        out, _ = self.rnn(self.enc(x))
        return self.head(out.mean(dim=1))


class R3D18(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        net.fc = nn.Linear(net.fc.in_features, 2)
        self.net = net

    def forward(self, x):  # (B,T,C,H,W) -> (B,C,T,H,W)
        return self.net(x.permute(0, 2, 1, 3, 4))


class Wrapper(nn.Module):
    """Gives baselines the same interface as the HF models (pixel_values -> .logits)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.net(pixel_values))


BUILDERS = {
    "cnn_lstm": (CNNLSTM, IMAGENET),
    "cnn_bigru": (CNNBiGRU, IMAGENET),
    "r3d18": (R3D18, KINETICS),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--model", choices=BUILDERS, required=True)
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Model: {args.model}")

    # same splits as train_transformers.py (same seed -> same val split)
    train_all = collect_videos(Path(args.root) / "train")
    test_s = collect_videos(Path(args.root) / "test")
    train_s, val_s = train_test_split(
        train_all, test_size=args.val_size,
        stratify=[l for _, l in train_all], random_state=args.seed)
    print(f"Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    cls, (mean, std) = BUILDERS[args.model]

    def loader(s, train):
        ds = SHVDataset(s, args.num_frames, 224, mean, std, train)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    train_loader, val_loader, test_loader = loader(train_s, True), loader(val_s, False), loader(test_s, False)

    model = Wrapper(cls()).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler() if (args.fp16 and device.type == "cuda") else None

    history, best = [], float("inf")
    best_path = out / "best_model.pt"
    for ep in range(1, args.epochs + 1):
        tl, ta, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        vl, va, _, _ = run_epoch(model, val_loader, device, criterion)
        scheduler.step(vl)
        history.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"Epoch {ep:02d} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), best_path)
            print("  -> saved best model")

    model.load_state_dict(torch.load(best_path, map_location=device))
    tl, ta, preds, gts = run_epoch(model, test_loader, device, criterion)
    cm = confusion_matrix(gts, preds)
    mf1 = f1_score(gts, preds, average="macro")
    print("\n===== TEST RESULTS =====")
    print(f"Loss {tl:.4f} | Accuracy {ta*100:.2f}% | Macro F1 {mf1:.3f}")
    print(cm)
    print(classification_report(gts, preds, target_names=CLASS_NAMES, digits=3))

    with open(out / "results.json", "w") as f:
        json.dump(dict(model=args.model, test_loss=tl, test_acc=ta, macro_f1=mf1,
                       confusion_matrix=cm.tolist(), history=history), f, indent=2)


if __name__ == "__main__":
    main()