"""
Classify a single video with a fine-tuned transformer model.
Must be in the same folder as train_transformers.py.

    python predict.py --video test_clip.mp4 --model timesformer
"""
import argparse
from pathlib import Path

import torch
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from train_transformers import MODEL_CONFIGS, read_frames

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--model", choices=MODEL_CONFIGS, required=True)
ap.add_argument("--model_path", default=None,
                help="Default: outputs/<model>/best_model")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = Path(args.model_path or Path("outputs") / args.model / "best_model")

proc = AutoImageProcessor.from_pretrained(path)
model = AutoModelForVideoClassification.from_pretrained(path).to(device).eval()

frames = read_frames(args.video, MODEL_CONFIGS[args.model]["num_frames"])
x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
x = v2.Resize((224, 224), antialias=True)(x).float() / 255.0
mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
x = ((x - mean) / std).unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.softmax(model(pixel_values=x).logits, dim=-1)[0].cpu()

print(f"\nVideo: {args.video}")
for i, p in enumerate(probs.tolist()):
    print(f"  {model.config.id2label[i]:<16} {p*100:6.2f}%")
pred = int(probs.argmax())
print(f"\nPrediction: {model.config.id2label[pred]}  (confidence {probs[pred]*100:.1f}%)")