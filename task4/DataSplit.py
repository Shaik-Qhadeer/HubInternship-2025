import os
import shutil
import random
from pathlib import Path

# Set paths
images_dir = Path("images")
labels_dir = Path("labels")
output_base = Path("dataset")
splits = {"train": 0.8, "val": 0.1, "test": 0.1}  # Adjust ratios as needed

# Collect all image files
image_files = sorted([f for f in images_dir.glob("*.jpg")])  # or .png if needed
random.shuffle(image_files)

# Split indices
n = len(image_files)
n_train = int(n * splits["train"])
n_val = int(n * splits["val"])
train_files = image_files[:n_train]
val_files = image_files[n_train:n_train + n_val]
test_files = image_files[n_train + n_val:]

split_map = {"train": train_files, "val": val_files, "test": test_files}

for split, files in split_map.items():
    img_out = output_base / split / "images"
    lbl_out = output_base / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path in files:
        # Copy image
        shutil.copy(img_path, img_out / img_path.name)
        # Copy label
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, lbl_out / label_path.name)
        else:
            print(f"Warning: No label found for {img_path.name}")

print("Dataset split complete!")