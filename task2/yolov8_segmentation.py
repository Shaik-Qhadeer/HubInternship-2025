from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n-seg.pt")  # or use yolov8s-seg.pt if downloaded

input_folder = "extracted_frames"
output_folder = "segmented_frames"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    frame = cv2.imread(input_path)
    results = model(frame)[0].plot()
    cv2.imwrite(output_path, results)

print("Segmentation completed.")
