import os

from ultralytics import YOLO

EPOCHS = int(os.environ.get("EPOCHS", 50))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 416))
BATCH = int(os.environ.get("BATCH", 8))
MODEL = os.environ.get("MODEL", "yolov8n.pt")
DATA = os.environ.get("DATA", "/workspace/data.yaml")


def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",
        project="/workspace/outputs",
        name="train",
        exist_ok=True,
        patience=20,
    )


if __name__ == "__main__":
    main()
