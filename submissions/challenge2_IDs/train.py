import sys
import time
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Find the directory where train.py is located
CURRENT_DIR = Path(__file__).resolve().parent

HACKATHON_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(HACKATHON_ROOT))

from base_model import ImageNetSubset
from submissions.challenge2_IDs.model import ModelArchitecture

OUTPUT = Path("weights.joblib")

# Matching evaluate.py dimensions
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Matching evaluate.py normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def rand_bbox(size, lam):
    """Generates random bounding box coordinates for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Choose a random center pixel
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    # Calculate the box boundaries
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    return bbx1, bby1, bbx2, bby2

def main():
    """
    Full training pipeline with validation tracking and timing.
    """
    # Set up device agnostic training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Define Training and Validation transforms
    train_transform = transforms.Compose([
        # --- 1. SPATIAL & GEOMETRIC ---
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.3),

        # --- 2. COLOR & LIGHTING ---
        # NEW: 20% chance to completely invert all colors (forces shape recognition)
        transforms.RandomInvert(p=0.2),
        
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2)
        ], p=0.4),

        # --- 3. BLUR (Camera Focus Vulnerability) ---
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ], p=0.3),

        # --- CONVERT TO MATH TENSOR ---
        transforms.ToTensor(),

        # --- 4. OCCLUSION (Random Erasing Vulnerability) ---
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.5, 2.0), value=0),

        # --- 5. NORMALIZE ---
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
    dataset_root = HACKATHON_ROOT / "dataset_split"

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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

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

            # --- CUTMIX LOGIC (50% Activation Probability) ---
            r = torch.rand(1).item()
            if r < 0.5:
                # 1. Generate a random mix ratio (lambda)
                lam = torch.rand(1).item()
                
                # 2. Shuffle the batch to get partner images
                rand_index = torch.randperm(inputs.size(0)).to(device)
                target_a = labels # Original labels
                target_b = labels[rand_index] # Partner labels
                
                # 3. Get the bounding box and physically cut-and-paste the pixels
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[rand_index, :, bby1:bby2, bbx1:bbx2]
                
                # 4. Adjust lambda based on the exact pixel area that was replaced
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                
                # 5. Forward pass and Mixed Loss calculation
                logits = model(inputs)
                loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
            else:
                # --- STANDARD LOGIC (For the other 50% of batches) ---
                logits = model(inputs)
                loss = criterion(logits, labels)
            # -------------------------------------------------

            loss.backward()
            optimizer.step()

            # --- RESTORED METRICS TRACKING ---
            train_loss += loss.item()
            train_total += labels.size(0)
            
            preds = logits.argmax(dim=1)
            if r < 0.5:
                # For CutMix, we grade the prediction against whichever image took up the majority of the space
                dominant_label = target_a if lam > 0.5 else target_b
                train_correct += (preds == dominant_label).sum().item()
            else:
                # Standard grading
                train_correct += (preds == labels).sum().item()
            # --------------------------------- 

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
        scheduler.step(avg_val_loss)
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