"""Run animal detection on an image, a folder of images, or a video.

Draws bounding boxes with class + confidence, and overlays a warning
banner on any frame/image where an animal is detected.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

WARNING_TEXT = "WARNING: Animal detected ahead"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


BIG_ANIMALS = {"boar", "deer", "bear", "cow", "horse", "elephant", "zebra", "giraffe"}

def get_danger_level(results, frame_shape):
    if len(results.boxes) == 0:
        return "SAFE: No object detected", (0, 255, 0)
        
    h, w = frame_shape[:2]
    frame_area = h * w
    
    max_danger_score = -1
    best_text = ""
    best_color = (0, 255, 0)
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area
        
        is_big = class_name in BIG_ANIMALS
        is_close = area_ratio > 0.05
        
        if is_big and is_close:
            score = 3
            text = "CRITICAL: BIG ANIMAL CLOSE"
            color = (0, 0, 255) # Red
        elif (is_big and not is_close) or (not is_big and is_close):
            score = 2
            text = "HIGH: ANIMAL AHEAD"
            color = (0, 165, 255) # Orange
        else:
            score = 1
            text = "LOW: SMALL ANIMAL FAR"
            color = (0, 255, 255) # Yellow
            
        if score > max_danger_score:
            max_danger_score = score
            best_text = text
            best_color = color
            
    return best_text, best_color

def annotate_frame(model, frame, conf):
    results = model.predict(frame, conf=conf, device="cpu", verbose=False)[0]
    annotated = frame.copy()
    
    text, bg_color = get_danger_level(results, frame.shape)
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    h, w = annotated.shape[:2]
    font_scale = max(0.5, w / 700)
    thickness = max(1, round(w / 250))
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = 10, h - 20
    cv2.rectangle(annotated, (x - 5, y - text_h - 10), (x + text_w + 5, y + baseline), bg_color, -1)
    
    text_color = (0, 0, 0) if bg_color in [(0, 255, 0), (0, 255, 255)] else (255, 255, 255)
    cv2.putText(
        annotated, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA,
    )
    return annotated, results


def run_video(model, source, output, conf):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    n_frames, n_detections = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, results = annotate_frame(model, frame, conf)
        if len(results.boxes) > 0:
            n_detections += 1
        writer.write(annotated)
        n_frames += 1

    cap.release()
    writer.release()
    print(f"Processed {n_frames} frames, {n_detections} with detections")
    print(f"Saved video to {output}")


def run_images(model, source, output_dir, conf, limit=None):
    src = Path(source)
    if src.is_file():
        paths = [src]
    else:
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if limit:
            paths = paths[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        annotated, results = annotate_frame(model, frame, conf)
        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), annotated)
        print(f"{p.name}: {len(results.boxes)} detection(s) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/workspace/outputs/train/weights/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to process")
    args = parser.parse_args()

    model = YOLO(args.weights)

    src = Path(args.source)
    if not src.is_absolute():
        src = Path("/workspace") / src

    output = args.output
    if not Path(output).is_absolute():
        output = str(Path("/workspace") / output)

    if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
        run_video(model, str(src), output, args.conf)
    else:
        run_images(model, str(src), output, args.conf, args.limit)


if __name__ == "__main__":
    main()
