FROM ultralytics/ultralytics:latest-cpu

WORKDIR /workspace

# Cache YOLOv8n base weights in the image so training doesn't need to
# re-download them every container run.
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install Flask Werkzeug yt-dlp
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
