import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HarassmentDataset(Dataset):

    def __init__(self, root_dir, num_frames=16, image_size=224):

        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size

        self.classes = {
            "Non_Harassment": 0,
            "Harassment": 1
        }

        self.video_paths = []
        self.labels = []

        for class_name, label in self.classes.items():

            class_folder = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.exists(class_folder):
                print("Folder not found:", class_folder)
                continue

            for filename in os.listdir(class_folder):

                if filename.lower().endswith(".mp4"):

                    video_path = os.path.join(
                        class_folder,
                        filename
                    )

                    self.video_paths.append(video_path)
                    self.labels.append(label)

        print("Dataset loaded:", len(self.video_paths), "videos")

    def __len__(self):

        return len(self.video_paths)

    def __getitem__(self, index):

        video_path = self.video_paths[index]
        label = self.labels[index]

        frames = self.load_video(video_path)

        # Convert list of NumPy arrays into one NumPy array first.
        # This is much faster than torch.tensor(list_of_arrays)
        frames = np.array(frames, dtype=np.float32)

        # Convert NumPy array to PyTorch tensor
        frames = torch.from_numpy(frames)

        # Change:
        # (frames, height, width, channels)
        #
        # to:
        # (frames, channels, height, width)

        frames = frames.permute(0, 3, 1, 2)

        # Normalize pixel values
        frames = frames / 255.0

        return frames, torch.tensor(
            label,
            dtype=torch.long
        )

    def load_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frames = []

        if total_frames <= 0:

            cap.release()

            raise RuntimeError(
                f"Could not read video: {video_path}"
            )

        # Select frames uniformly
        frame_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int
        )

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if success:

                # OpenCV uses BGR.
                # Convert BGR -> RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Resize frame
                frame = cv2.resize(
                    frame,
                    (
                        self.image_size,
                        self.image_size
                    )
                )

                frames.append(frame)

        cap.release()

        # If some frames could not be read,
        # duplicate the last available frame.

        while len(frames) < self.num_frames:

            if len(frames) > 0:

                frames.append(frames[-1])

            else:

                frames.append(
                    np.zeros(
                        (
                            self.image_size,
                            self.image_size,
                            3
                        ),
                        dtype=np.uint8
                    )
                )

        return frames


if __name__ == "__main__":

    dataset_path = (
        r"D:\minor25oct\Sexual_Harassment_Dataset"
        r"\Sexual_Harassment_Dataset\train"
    )

    dataset = HarassmentDataset(
        dataset_path
    )

    video, label = dataset[0]

    print(
        "Video tensor shape:",
        video.shape
    )

    print(
        "Label:",
        label.item()
    )