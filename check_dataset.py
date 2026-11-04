import os

DATASET_PATH = r"D:\minor25oct\Sexual_Harassment_Dataset\Sexual_Harassment_Dataset"

for split in ["train", "test"]:

    print("\n==============================")
    print(split.upper())
    print("==============================")

    for category in ["Harassment", "Non_Harassment"]:

        folder = os.path.join(DATASET_PATH, split, category)

        if not os.path.exists(folder):
            print("Folder not found:", folder)
            continue

        videos = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".mp4")
        ]

        print(category, ":", len(videos), "videos")