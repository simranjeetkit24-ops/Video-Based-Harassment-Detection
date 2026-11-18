from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np, torch, av

name = "facebook/timesformer-base-finetuned-k400"
proc = AutoImageProcessor.from_pretrained(name)
model = TimesformerForVideoClassification.from_pretrained(name)

def read_frames(path, n=8):
    c = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]

frames = read_frames("clip.mp4")
inputs = proc(frames, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(model.config.id2label[logits.argmax(-1).item()])