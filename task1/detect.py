import urllib.request
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Download the image
url = 'https://ultralytics.com/images/bus.jpg'
urllib.request.urlretrieve(url, 'bus.jpg')

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection
results = model("bus.jpg", save=True)

# Print results
for r in results:
    boxes = r.boxes
    names = r.names
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"{names[cls]} - {conf:.2f}")



# Correct output path handling
output_path = Path(results[0].save_dir) / "bus.jpg"
img = cv2.imread(str(output_path))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Detection Output")
plt.show()