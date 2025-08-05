from ultralytics import YOLO
from pathlib import Path
import os

# Load segmentation model (YOLOv8n-seg by default)
model = YOLO("yolov8n-seg.pt")

# Input and output directories
input_folder = Path("images")
output_folder = Path("segmented")
output_folder.mkdir(exist_ok=True)

# List all image files in the input directory
image_files = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png")) + list(input_folder.glob("*.jpeg"))

# Process each image
for image_path in image_files:
    print(f"Segmenting: {image_path.name}")
    results = model.predict(source=str(image_path), save=True, save_txt=False, conf=0.3)

    # Move output image to the 'segmented/' folder
    saved_dir = Path(results[0].save_dir)
    segmented_image = saved_dir / image_path.name
    if segmented_image.exists():
        segmented_image.rename(output_folder / image_path.name)

print(" Segmentation completed and results saved in 'segmented/' folder.")
