# In this file we perform basic EDA on the image dataset.
# EDA = Exploratory Data Analysis.
# The goal is to understand the dataset before training the model

from pathlib import Path
import pandas as pd

#---------------------- Df basics -------------------------

# Define the path to the folder that contains all image class folders.
DATA_ROOT = Path("train")

# Define all image file extensions that we want to search for.
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

# Each item in this list will represent one image and its label.
rows = []

# Loop over all folders inside the train directory.
# Each folder name is treated as the class label.
for class_dir in sorted(DATA_ROOT.iterdir()):

    # Check that the current item is actually a folder and not a file.
    if class_dir.is_dir():
         #Save the folder name as the class name.
        class_name = class_dir.name 

        # Create an empty list for all image paths in the current class folder.
        image_paths = []

        # Search for images with each possible extension.
        for ext in IMAGE_EXTENSIONS:
            # Add all image paths with the current extension to image_paths.
            image_paths.extend(class_dir.glob(ext)) 

        # Loop over all images found in the current class folder.
        for image_path in image_paths:
            # Each row contains the image path and the matching label.
            rows.append({"image_path": str(image_path),"label": class_name})

# Convert the list of rows into a pandas DataFrame.
# The DataFrame will have two columns: image_path and label.
df = pd.DataFrame(rows)

# Print the total number of images in the dataset.
print("\nTotal number of images:", len(df))

# Print all unique class names in the dataset.
print("\nClasses:")
print(df["label"].unique())

# Print the number of different classes.
print("\nNumber of classes:", df["label"].nunique())

# Print the number of images in each class.
print("\nNumber of images per class:")
print(df["label"].value_counts())


#---------------------- Df Graphs -------------------------





