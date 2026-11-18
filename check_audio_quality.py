import av, glob, os
import numpy as np
from collections import defaultdict

root = r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset"
files = glob.glob(root + r"\*\*\*.mp4")
stats = defaultdict(list)   # class -> list of RMS values

for f in files:
    cls = os.path.basename(os.path.dirname(f))
    try:
        c = av.open(f)
        if not c.streams.audio:
            stats[cls].append(0.0)
            continue
        chunks = []
        for fr in c.decode(audio=0):
            a = fr.to_ndarray()
            if a.dtype == np.int16:
                a = a / 32768.0
            chunks.append(a.astype(np.float32).ravel())
        c.close()
        x = np.concatenate(chunks) if chunks else np.zeros(1)
        stats[cls].append(float(np.sqrt(np.mean(x ** 2))))
    except Exception as e:
        print("error", f, e)

for cls, v in stats.items():
    v = np.array(v)
    print(f"{cls}: {len(v)} clips | silent (RMS<0.005): {(v < 0.005).sum()} | median RMS: {np.median(v):.4f}")