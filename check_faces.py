import cv2, glob, os
import numpy as np
from collections import defaultdict

root = r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset"
files = glob.glob(root + r"\*\*\*.mp4")
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

has_face = defaultdict(list)   # class -> 1/0 per clip
face_frac = defaultdict(list)  # class -> face height / frame height

for f in files:
    cls = os.path.basename(os.path.dirname(f))
    cap = cv2.VideoCapture(f)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found, best = 0, 0.0
    for i in np.linspace(0, max(n - 1, 0), 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(g, 1.1, 4, minSize=(24, 24))
        if len(faces):
            found += 1
            best = max(best, max(h for (_, _, _, h) in faces) / fr.shape[0])
    cap.release()
    has_face[cls].append(1 if found else 0)
    if found:
        face_frac[cls].append(best)

for cls in has_face:
    v = has_face[cls]
    ff = np.median(face_frac[cls]) if face_frac[cls] else 0
    print(f"{cls}: {len(v)} clips | face found in {sum(v)} ({100*sum(v)/len(v):.0f}%) | median face height: {ff:.2f} of frame")