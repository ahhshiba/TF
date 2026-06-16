import os
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

app = Flask(__name__, template_folder='../templates')

UPLOAD_FOLDER = Path('/workspace/outputs/uploads')
RESULTS_FOLDER = Path('/workspace/outputs/results')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Try to load the best model if trained, else fallback to yolov8n
weights_path = "/workspace/outputs/train/weights/best.pt"
if not os.path.exists(weights_path):
    print("Warning: best.pt not found. Using base yolov8n.pt")
    weights_path = "yolov8n.pt"

try:
    model = YOLO(weights_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = YOLO("yolov8n.pt")

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
        is_close = area_ratio > 0.05  # >5% of frame
        
        if is_big and is_close:
            score = 3
            text = "CRITICAL: BIG ANIMAL CLOSE"
            color = (0, 0, 255) # Red
        elif (is_big and not is_close) or (not is_big and is_close):
            score = 2
            text = "HIGH: ANIMAL AHEAD"
            color = (0, 165, 255) # Orange (BGR)
        else:
            score = 1
            text = "LOW: SMALL ANIMAL FAR"
            color = (0, 255, 255) # Yellow (BGR)
            
        if score > max_danger_score:
            max_danger_score = score
            best_text = text
            best_color = color
            
    return best_text, best_color

def annotate_frame(frame, conf=0.4):
    results = model.predict(frame, conf=conf, device="cpu", verbose=False)[0]
    annotated = frame.copy()
    
    text, bg_color = get_danger_level(results, frame.shape)
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    h, w = annotated.shape[:2]
    font_scale = max(0.5, w / 700)
    thickness = max(1, round(w / 250))
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = 10, h - 20
    cv2.rectangle(annotated, (x - 5, y - text_h - 10), (x + text_w + 5, y + baseline), bg_color, -1)
    
    text_color = (0, 0, 0) if bg_color in [(0, 255, 0), (0, 255, 255)] else (255, 255, 255)
    cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return annotated

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        
        filename = secure_filename(file.filename)
        input_path = UPLOAD_FOLDER / filename
        file.save(input_path)
        
        ext = input_path.suffix.lower()
        output_filename = f"result_{filename}"
        output_path = RESULTS_FOLDER / output_filename
        
        if ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(input_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            while True:
                ok, frame = cap.read()
                if not ok: break
                annotated = annotate_frame(frame)
                writer.write(annotated)
                
            cap.release()
            writer.release()
            
            # Convert to web-compatible h264
            web_output_path = RESULTS_FOLDER / f"web_{output_filename}"
            os.system(f"ffmpeg -y -i {output_path} -vcodec libx264 -f mp4 {web_output_path} > /dev/null 2>&1")
            
            return render_template('index.html', video_url=url_for('get_result', filename=f"web_{output_filename}"))
            
        elif ext in IMAGE_EXTS:
            frame = cv2.imread(str(input_path))
            annotated = annotate_frame(frame)
            cv2.imwrite(str(output_path), annotated)
            return render_template('index.html', image_url=url_for('get_result', filename=output_filename))
            
    return render_template('index.html')

@app.route('/test_sample', methods=['POST'])
def test_sample():
    input_path = Path("/workspace/demo_input/github_sample.mp4")
    if not input_path.exists():
        return "Sample video not found", 404
        
    output_filename = "result_github_sample.mp4"
    output_path = RESULTS_FOLDER / output_filename
    
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        annotated = annotate_frame(frame)
        writer.write(annotated)
        
    cap.release()
    writer.release()
    
    web_output_path = RESULTS_FOLDER / f"web_{output_filename}"
    os.system(f"ffmpeg -y -i {output_path} -vcodec libx264 -f mp4 {web_output_path} > /dev/null 2>&1")
    
    return render_template('index.html', video_url=url_for('get_result', filename=f"web_{output_filename}"))

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(str(RESULTS_FOLDER), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
