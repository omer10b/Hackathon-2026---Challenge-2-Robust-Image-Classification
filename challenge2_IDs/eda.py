# In this file we perform basic EDA on the image dataset.
# EDA = Exploratory Data Analysis.
# The goal is to understand the dataset before training the model

# ------------------------- load libraries -------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from matplotlib import colors

# ------------------------- Dataset settings -------------------------

# Define the path to the folder that contains all image class folders.
DATA_ROOT = Path("train")

# Define all image file extensions that we want to search for.
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


#------------------------- Df -------------------------

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

# ------------------------- Basic dataset information -------------------------

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


#------------------------- Df Graphs -------------------------

# -------------------------  Sample -------------------------

# Number of images to sample from each class.
SAMPLE_PER_CLASS = 300

# Create a balanced sample from the dataset.
sample_parts = []

for label in df["label"].unique():
    label_df = df[df["label"] == label]

    sampled_label_df = label_df.sample(n=min(len(label_df), SAMPLE_PER_CLASS),random_state=42)
    sample_parts.append(sampled_label_df)
sample_df = pd.concat(sample_parts, ignore_index=True)

# -------------------------  color palette -------------------------

# Use a fire-like color map.
fire_cmap = plt.cm.inferno

# Number of labels.
label_counts = df["label"].value_counts()

# Create fire colors according to the number of classes.
fire_colors = fire_cmap(np.linspace(0.25, 0.95, len(label_counts)))


# ------------------------- Images per class -------------------------

# Create a bar chart that shows the number of images per class.
plt.figure(figsize=(12, 6))
plt.bar(label_counts.index, label_counts.values, color=fire_colors, edgecolor="black")
# Add titles and labels to the graph.
plt.title("Number of Images per Class")
plt.xlabel("Class")
plt.ylabel("Number of Images")

# Rotate class names so they will be readable.
plt.xticks(rotation=90)
# Make sure everything fits nicely inside the figure.
plt.tight_layout()
# Display the graph.
plt.show()


# ------------------------- One example image per class -------------------------

# Number of unique classes.
classes = sorted(df["label"].unique())

# Calculate the number of rows and columns needed for the grid.
num_classes = len(classes)
cols = 6
rows = math.ceil(num_classes / cols)

#` Create a figure with the specified size.`
plt.figure(figsize=(12, 2.5 * rows))

# Loop over all classes and display one image from each class.
for i, label in enumerate(classes):
    # Select the first image that belongs to the current class.
    image_path = df[df["label"] == label]["image_path"].iloc[0]

    # Open the image.
    image = Image.open(image_path).convert("RGB")

    # Display image in subplot.
    plt.subplot(rows, cols, i + 1)
    plt.imshow(image)
    plt.title(label, fontsize=8)
    plt.axis("off")

# Make sure everything fits inside the figure.
plt.tight_layout()
# Display the graph.
plt.show() 
 
# ------------------------- Brightness Graph -------------------------

#Calculate brightness 

brightness_values = []
for image_path in sample_df["image_path"]:
    with Image.open(image_path) as image:
        image = image.convert("L").resize((128, 128))
        image_array = np.array(image) / 255.0

    brightness_values.append(image_array.mean())

sample_df["brightness"] = brightness_values


brightness_per_class = sample_df.groupby("label")["brightness"].mean()

# Normalize the brightness values to the range [0, 1]
norm = colors.Normalize(vmin=brightness_per_class.min(),vmax=brightness_per_class.max())
normalized_brightness = norm(brightness_per_class.values)

# Use the normalized brightness values to choose colors from the fire palette
bar_colors = fire_cmap(normalized_brightness)

# Create the bar chart
plt.figure(figsize=(14, 6))
plt.bar(brightness_per_class.index,brightness_per_class.values,color=bar_colors,edgecolor="black")
plt.title("Average Brightness per Class - Sample")
plt.xlabel("Class")
plt.ylabel("Average Brightness")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show(block=True) 

# ------------------------- Average color per class Graph -------------------------

#Calculate average RGB for each image

red_means = []
green_means = []
blue_means = []

for image_path in sample_df["image_path"]:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((128, 128))
        image_array = np.array(image) / 255.0

    red_means.append(image_array[:, :, 0].mean())
    green_means.append(image_array[:, :, 1].mean())
    blue_means.append(image_array[:, :, 2].mean())

sample_df["red_mean"] = red_means
sample_df["green_mean"] = green_means
sample_df["blue_mean"] = blue_means


#Calculate average color per class 
rgb_per_class = sample_df.groupby("label")[["red_mean", "green_mean", "blue_mean"]].mean()

# Convert each class average RGB into a color tuple for matplotlib.
bar_colors = [(row["red_mean"], row["green_mean"], row["blue_mean"])for _, row in rgb_per_class.iterrows()]

bar_heights = [1] * len(rgb_per_class)
plt.figure(figsize=(14, 6))
# create a bar chart with the average color for each class
plt.bar(rgb_per_class.index,bar_heights,color=bar_colors,edgecolor="black")

# Add titles and labels to the graph.
plt.title("Average Color per Class")
plt.xlabel("Class")
plt.ylabel("Color Display")
plt.xticks(rotation=90)
plt.yticks([])      
plt.tight_layout()
plt.show(block=True) 


# ------------------------- Average contrast per class Graph -------------------------

# Calculate contrast
contrast_values = []

for image_path in sample_df["image_path"]:
    with Image.open(image_path) as image:
        # Convert to grayscale and resize for faster calculation
        image = image.convert("L").resize((128, 128))
        image_array = np.array(image) / 255.0

    # Standard deviation of pixel values = contrast
    contrast_values.append(image_array.std())

sample_df["contrast"] = contrast_values

# Average contrast per class
contrast_per_class = sample_df.groupby("label")["contrast"].mean()

# Optional: sort classes by contrast
contrast_per_class = contrast_per_class.sort_values()

# Normalize contrast values to range [0, 1]
norm = colors.Normalize(vmin=contrast_per_class.min(),vmax=contrast_per_class.max())

normalized_contrast = norm(contrast_per_class.values)

gray_values = 0.9 - 0.8 * normalized_contrast
bar_colors = [(g, g, g) for g in gray_values]

plt.figure(figsize=(14, 6))
plt.bar(contrast_per_class.index,contrast_per_class.values,color=bar_colors,edgecolor="black")

plt.title("Average Contrast per Class")
plt.xlabel("Class")
plt.ylabel("Average Contrast")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show(block=True)

