from ultralytics import YOLO
from pathlib import Path

# === CONFIG ===
MODEL_ARCH = "yolov8n.pt" 
DATA_YAML = "data.yaml"     
PROJECT_DIR = Path("outputs")
TRAIN_NAME = "dicefaces"
IMG_SIZE = 640
EPOCHS = 150
DEVICE = 0  

def main():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    model = YOLO(MODEL_ARCH)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(PROJECT_DIR),
        name=TRAIN_NAME,
        device=DEVICE,
        exist_ok=True
    )

    best_weights = PROJECT_DIR / TRAIN_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError("Training completed but best.pt not found.")

    model = YOLO(str(best_weights))
    print(f"Loaded trained weights: {best_weights}")

    print("Validating on val set...")
    model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        split="val",
        device=DEVICE
    )

    # Custom prediction visualization for validation set
    val_images_dir = Path("dataset/val/images")
    if val_images_dir.exists():
        print("Running predictions on validation images...")
        val_results = model.predict(
            source=str(val_images_dir),
            imgsz=IMG_SIZE,
            conf=0.25,
            project=str(PROJECT_DIR),
            name="pred_val_images",
            exist_ok=True,
            device=DEVICE,
            save=False,
            stream=True
        )

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        import os

        class_names = model.model.names if hasattr(model.model, "names") else []
        num_classes = len(class_names)
        random.seed(42)
        colors = [tuple(random.choices(range(256), k=3)) for _ in range(num_classes)]
        colors = [(r/255, g/255, b/255) for r, g, b in colors]

        pred_val_dir = PROJECT_DIR / "pred_val_images_custom"
        pred_val_dir.mkdir(parents=True, exist_ok=True)

        for result in val_results:
            im = result.orig_img
            boxes = result.boxes
            fig, ax = plt.subplots(1)
            ax.imshow(im)
            for box in boxes:
                cls = int(box.cls[0]) if hasattr(box, "cls") else 0
                color = colors[cls % num_classes] if num_classes > 0 else (1,0,0)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            img_name = os.path.basename(result.path)
            plt.savefig(pred_val_dir / img_name, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        print(f"Predictions with colored boxes saved in: {pred_val_dir}")
    else:
        print("Validation images folder not found.")

    print("Validating on test set...")
    model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        split="test",
        device=DEVICE
    )

    # Prepare legend once for both val and test
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random
    import os

    class_names = model.model.names if hasattr(model.model, "names") else []
    num_classes = len(class_names)
    random.seed(42)
    colors = [tuple(random.choices(range(256), k=3)) for _ in range(num_classes)]
    colors = [(r/255, g/255, b/255) for r, g, b in colors]

    # Save legend as a separate image
    legend_fig, legend_ax = plt.subplots(figsize=(2, max(2, num_classes * 0.3)))
    legend_handles = [
        patches.Patch(edgecolor=color, facecolor='none', label=class_names[i] if class_names else f"Class {i}", linewidth=2)
        for i, color in enumerate(colors)
    ]
    legend_ax.legend(handles=legend_handles, loc='center', fontsize='small', title='Classes')
    legend_ax.axis('off')
    legend_path = PROJECT_DIR / "legend.png"
    plt.savefig(legend_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(legend_fig)
    print(f"Legend saved as: {legend_path}")

    test_images_dir = Path("dataset/test/images")
    if test_images_dir.exists():
        print("Running predictions on test images...")
        results = model.predict(
            source=str(test_images_dir),
            imgsz=IMG_SIZE,
            conf=0.25,
            project=str(PROJECT_DIR),
            name="pred_test_images",
            exist_ok=True,
            device=DEVICE,
            save=False,  # We'll handle saving ourselves
            stream=True  # To process results one by one
        )

        pred_dir = PROJECT_DIR / "pred_test_images_custom"
        pred_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            im = result.orig_img
            boxes = result.boxes
            fig, ax = plt.subplots(1)
            ax.imshow(im)
            # Draw boxes without labels or legend
            for box in boxes:
                cls = int(box.cls[0]) if hasattr(box, "cls") else 0
                color = colors[cls % num_classes] if num_classes > 0 else (1,0,0)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            img_name = os.path.basename(result.path)
            plt.savefig(pred_dir / img_name, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        print(f"Predictions with colored boxes saved in: {pred_dir}")
    else:
        print("Test images folder not found.")

    print("Training, validation, and prediction complete. Results are in:", PROJECT_DIR.resolve())

if __name__ == "__main__":
    main()
