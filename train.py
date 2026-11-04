import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import HarassmentDataset


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. DATASET
# ==========================================

DATASET_PATH = (
    r"D:\minor25oct\Sexual_Harassment_Dataset"
    r"\Sexual_Harassment_Dataset\train"
)

dataset = HarassmentDataset(
    DATASET_PATH,
    num_frames=16,
    image_size=224
)

print("Total videos:", len(dataset))


# ==========================================
# 3. TRAIN / VALIDATION SPLIT
# ==========================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training videos:", len(train_dataset))
print("Validation videos:", len(val_dataset))


# ==========================================
# 4. DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. SIMPLE CNN MODEL
# ==========================================

class SimpleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            128,
            2
        )


    def forward(self, x):

        # Input:
        # batch, frames, channels, height, width

        batch_size, frames, channels, height, width = x.shape

        # Treat every frame as an individual image

        x = x.reshape(
            batch_size * frames,
            channels,
            height,
            width
        )

        x = self.features(x)

        x = self.pool(x)

        x = x.view(
            batch_size * frames,
            -1
        )

        x = self.classifier(x)

        # Restore frame dimension

        x = x.view(
            batch_size,
            frames,
            2
        )

        # Average predictions across frames

        x = x.mean(dim=1)

        return x


# ==========================================
# 6. CREATE MODEL
# ==========================================

model = SimpleCNN().to(device)

print(model)


# ==========================================
# 7. LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 8. TRAINING
# ==========================================

epochs = 10

best_val_accuracy = 0.0


for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for videos, labels in train_loader:

        videos = videos.to(device)
        labels = labels.to(device)


        # Clear previous gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(videos)


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Loss

        running_loss += loss.item()


        # Predictions

        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * correct / total
    )


    # ======================================
    # VALIDATION AFTER EACH EPOCH
    # ======================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    # Average loss

    epoch_loss = (
        running_loss /
        len(train_loader)
    )


    print(
        f"\nEpoch [{epoch + 1}/{epochs}]"
    )

    print(
        f"Training Loss: {epoch_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            "Best model saved!"
        )


# ==========================================
# 9. FINAL RESULT
# ==========================================

print("\n==============================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Training completed."
)

print(
    "Model saved as: best_model.pth"
)

print("==============================")