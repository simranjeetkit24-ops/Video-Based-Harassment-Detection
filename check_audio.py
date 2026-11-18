import av, glob
files = glob.glob(r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset\*\*\*.mp4")
with_audio = sum(1 for f in files if av.open(f).streams.audio)
print(with_audio, "of", len(files), "videos have audio")