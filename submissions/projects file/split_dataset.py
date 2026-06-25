
"""
Split the original dataset into train, validation and test sets.
The split is performed independently for each class in order to
preserve the original class distribution (stratified split).

Split ratios:
- Train: 75%
- Validation: 10%
- Test: 15%

The random seed is fixed to ensure reproducibility.
"""

# ---------------------------------------------------
# Imports
# ---------------------------------------------------

from pathlib import Path
import random
import shutil


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

SOURCE_DIR = Path("train")         # Folder containing the original dataset
OUTPUT_DIR = Path("dataset_split")  # Folder that will contain the new split

# ---------------------------------------------------
# Split ratios
# ---------------------------------------------------

TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

# Fixed random seed for reproducibility

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG"}

random.seed(SEED)

# Output folders
train_dir = OUTPUT_DIR / "train"
val_dir = OUTPUT_DIR / "validation"
test_dir = OUTPUT_DIR / "test"

# ---------------------------------------------------
# Safety checks
# ---------------------------------------------------

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

# Prevent accidentally overwriting an existing split
if OUTPUT_DIR.exists():
    raise RuntimeError(f"{OUTPUT_DIR} already exists. Delete it first if you want to recreate the split.")

# ---------------------------------------------------
# Iterate over every class folder
# ---------------------------------------------------
for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    # Collect all images from the current class
    images = [p for p in class_dir.iterdir()if p.is_file() and p.suffix in IMAGE_EXTENSIONS]

    # Sort for consistency and then shuffle randomly
    images = sorted(images)
    random.shuffle(images)

    # Number of images in this class
    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

# ---------------------------------------------------
# Copy images into their corresponding folders
# ---------------------------------------------------
    for split_name, split_images in [("train", train_images),("validation", val_images),("test", test_images),]:
        out_class_dir = OUTPUT_DIR / split_name / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            shutil.copy2(img_path, out_class_dir / img_path.name)

    print(f"{class_dir.name}: "f"train={len(train_images)}, "f"validation={len(val_images)}, "f"test={len(test_images)}"
    )

# ---------------------------------------------------
# Finished
# ---------------------------------------------------
print("\nDone.")
print("\nDataset split completed successfully!")
