from ultralytics import YOLO

# Load a pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="african-wildlife.yaml",
    epochs=20,
    imgsz=640,
    batch=16
)
