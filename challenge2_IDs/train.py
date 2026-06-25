import time
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from base_model import ImageNetSubset
from model import ModelArchitecture

# Find the directory where train.py is located
CURRENT_DIR = Path(__file__).resolve().parent

# Go up two levels to reach the hackathon root folder
HACKATHON_ROOT = CURRENT_DIR.parent.parent

OUTPUT = Path("weights.joblib")

# Matching evaluate.py dimensions
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Matching evaluate.py normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    """
    Full training pipeline with validation tracking and timing.
    """
    # Set up device agnostic training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Define Training and Validation transforms
    train_transform = transforms.Compose([
        # 1. Randomly resize and crop (forces model to recognize parts of objects)
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),

        # 2. Randomly flip left/right (mirrors the image)
        transforms.RandomHorizontalFlip(p=0.5),

        # 3. Slight rotations (fixes perfectly upright biases)
        transforms.RandomRotation(degrees=15),

        # 4. Color Jitter (simulates different lighting/camera qualities)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 2. Load datasets and initialize DataLoaders
    # This perfectly matches your dataset_split folder structure
    dataset_root = HACKATHON_ROOT / "train_set" / "dataset_split"

    print(f"Initializing datasets from: {dataset_root}")

    # Builds: .../train_set/dataset_split/train
    train_dataset = ImageNetSubset(
        root=dataset_root,
        split="train",
        transform=train_transform
    )

    # Builds: .../train_set/dataset_split/validation
    val_dataset = ImageNetSubset(
        root=dataset_root,
        split="validation",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # 3. Create your model
    print("Initializing ModelArchitecture...")
    model = ModelArchitecture(num_classes=20).to(device)

    # 4. Set up Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop
    print("Starting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                logits = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTraining completed in {minutes}m {seconds}s")

    # 6. Save trained model weights to weights.joblib
    model.to("cpu")
    joblib.dump(model.state_dict(), OUTPUT)
    print(f"Saved trained weights to {OUTPUT}")


if __name__ == "__main__":
    main()