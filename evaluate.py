"""
Expected submissions layout:
  submissions/
    team_a/
      train.py
      model.py
      predict.py
      weights.joblib
    team_b/
      train.py
      model.py
      predict.py
      weights.joblib

Run:
  python evaluate.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from labels import (HF_INDEX_TO_NAME,HF_INDEX_TO_IDX,TARGET_HF_INDICES,)

# ── editable ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path("dataset_split")   # contains train/ and validation/
SUBMISSIONS_DIR = Path("submissions")
BATCH_SIZE = 64
WEIGHTS_FILENAME = "weights.joblib"
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ImageNetSubset(Dataset):
    """Loads the 20 target classes from data/dataset/validation."""

    def __init__(self, root: Path, split: str = "validation", transform=None):
        self.transform = transform
        self.samples = []

        split_root = root / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Validation folder not found: {split_root}\n"
                f"Expected structure: {root}/validation/<class_name>/*.jpg"
            )

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

            local_idx = HF_INDEX_TO_IDX[hf_idx]

            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, local_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_test_sets():
    # 1. Clean Baseline
    clean_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # 2. Trained (Seen Manipulations - Light flip & jitter)
    seen_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # 3. Trained (Blur - Now part of the training curriculum)
    blur_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # 4. Trained (Erasing - Now part of the training curriculum)
    erase_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.5, 2.0), value=0),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # 5. Trained (Extreme Color - Now part of the training curriculum)
    color_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # --- NEW UNSEEN STRESS TESTS ---

    # 6. Untrained (Perspective - Geometric warping)
    perspective_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # 7. Untrained (Grayscale - Edge/Shape reliance)
    grayscale_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.RandomGrayscale(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # 8. Untrained (Solarize - Highlight inversion)
    solarize_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.RandomSolarize(threshold=192.0, p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # Package them all up into a dictionary of DataLoaders
    return {
        "Clean Baseline": DataLoader(ImageNetSubset(DATA_ROOT, "validation", clean_transform), batch_size=BATCH_SIZE),
        "Trained (Seen)": DataLoader(ImageNetSubset(DATA_ROOT, "validation", seen_transform), batch_size=BATCH_SIZE),
        "Trained (Blur)": DataLoader(ImageNetSubset(DATA_ROOT, "validation", blur_transform), batch_size=BATCH_SIZE),
        "Trained (Erasing)": DataLoader(ImageNetSubset(DATA_ROOT, "validation", erase_transform), batch_size=BATCH_SIZE),
        "Trained (Extreme Color)": DataLoader(ImageNetSubset(DATA_ROOT, "validation", color_transform), batch_size=BATCH_SIZE),
        "Untrained (Perspective)": DataLoader(ImageNetSubset(DATA_ROOT, "validation", perspective_transform), batch_size=BATCH_SIZE),
        "Untrained (Grayscale)": DataLoader(ImageNetSubset(DATA_ROOT, "validation", grayscale_transform), batch_size=BATCH_SIZE),
        "Untrained (Solarize)": DataLoader(ImageNetSubset(DATA_ROOT, "validation", solarize_transform), batch_size=BATCH_SIZE)
    }

# ── submission loading ────────────────────────────────────────────────────────

def load_submission(team_dir: Path):
    predict_path = team_dir / "predict.py"
    model_path = team_dir / "model.py"
    weights_path = team_dir / WEIGHTS_FILENAME

    if not predict_path.exists():
        raise FileNotFoundError(f"Missing predict.py in {team_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.py in {team_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {WEIGHTS_FILENAME} in {team_dir}")

    # So predict.py can do: from model import ModelArchitecture
    sys.path.insert(0, str(team_dir))

    # Important when grading multiple teams:
    # prevents Python from reusing a previous team's model.py
    sys.modules.pop("model", None)

    try:
        spec = importlib.util.spec_from_file_location(
            f"{team_dir.name}_predict",
            predict_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import predict.py from {team_dir}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Model"):
            raise AttributeError(f"predict.py in {team_dir} must define a class named Model")

        model = module.Model()
        model.load(str(weights_path))

    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)

    return model


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    correct = 0
    total   = 0
    for x, y in loader:
        preds    = model.predict(x)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Preparing test set...")
    loader = load_test_set()

    team_dirs = sorted(d for d in SUBMISSIONS_DIR.iterdir() if d.is_dir())
    if not team_dirs:
        print(f"No submissions found in {SUBMISSIONS_DIR}/")
        sys.exit(1)

    results = []
    for team_dir in team_dirs:
        print(f"Evaluating {team_dir.name}...", end=" ", flush=True)
        try:
            model = load_submission(team_dir)
            acc   = evaluate(model, loader)
            results.append((team_dir.name, acc))
            print(f"accuracy: {acc:.4f}")
        except Exception as e:
            print(f"FAILED — {e}")
            results.append((team_dir.name, None))

    print("\n--- Leaderboard ---")
    ranked = sorted((r for r in results if r[1] is not None), key=lambda r: r[1], reverse=True)
    for rank, (team, acc) in enumerate(ranked, start=1):
        print(f"  {rank}. {team:<20} {acc:.4f}")
    for team, acc in results:
        if acc is None:
            print(f"  --  {team:<20} FAILED")


if __name__ == "__main__":
    main()
