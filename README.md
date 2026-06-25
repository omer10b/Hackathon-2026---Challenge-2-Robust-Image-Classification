# Robust Image Classification - Baseline Model

## Overview
This project implements a custom Convolutional Neural Network (CNN) for the "Hackathon 2026 - Challenge 2: Robust Image Classification".
The objective is to classify images into 20 classes while improving robustness against common visual variations such as changes in object position, orientation, lighting conditions, and color distribution.
---

## Dataset Split
The original dataset was divided into three subsets using a fixed random seed (42) to ensure reproducibility while preserving the class distribution.
- **Training:** 75%
- **Validation:** 10%
- **Test:** 15%
Each class is split independently.

---

## Model Architecture
The model is a custom CNN implemented from scratch without using pretrained networks.
Architecture:
```
Input (224×224×3)
↓
Conv(32) + BatchNorm + ReLU + MaxPool
↓
Conv(64) + BatchNorm + ReLU + MaxPool
↓
Conv(128) + BatchNorm + ReLU + MaxPool
↓
Conv(256) + BatchNorm + ReLU + MaxPool
↓
Conv(512) + BatchNorm + ReLU + MaxPool
↓
Adaptive Average Pooling
↓
Linear(512 → 256)
↓
ReLU
↓
Dropout (0.2)
↓
Linear(256 → 20)
```

The network predicts logits for the 20 target classes.

---

## Training
Training configuration:
- Optimizer: Adam
- Learning Rate: 1e-3
- Loss Function: CrossEntropyLoss
- Epochs: 20
- Batch Size: 32
---

## Data Augmentation
To improve generalization and robustness, the following augmentations are applied during training:
- Random Resized Crop
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter
    - Brightness
    - Contrast
    - Saturation

---
