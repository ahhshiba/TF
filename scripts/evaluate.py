import os

from ultralytics import YOLO

WEIGHTS = os.environ.get("WEIGHTS", "/workspace/outputs/train/weights/best.pt")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 416))
DATA = os.environ.get("DATA", "/workspace/data.yaml")


def main():
    model = YOLO(WEIGHTS)
    metrics = model.val(
        data=DATA,
        split="test",
        imgsz=IMG_SIZE,
        device="cpu",
        project="/workspace/outputs",
        name="evaluate",
        exist_ok=True,
    )

    print("\n=== Test set metrics ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")

    print("\nPer-class mAP50:")
    for idx, name in metrics.names.items():
        print(f"  {name:10s}: {metrics.box.maps[idx]:.4f}")

    with open("/workspace/outputs/evaluate/metrics.txt", "w") as f:
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n")
        f.write("\nPer-class mAP50:\n")
        for idx, name in metrics.names.items():
            f.write(f"  {name}: {metrics.box.maps[idx]:.4f}\n")


if __name__ == "__main__":
    main()
