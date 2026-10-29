"""FastAPI backend: find signatures (YOLO11) and tell whose signature it is (optional)."""
import base64
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

try:
    from backend.utils import crop_box, draw_detections, identify
except ImportError:  # started from inside the backend folder
    from utils import crop_box, draw_detections, identify

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "weights" / "best.pt"))
PEOPLE_PATH = Path(os.getenv("PEOPLE_MODEL_PATH", ROOT / "weights" / "people_cls.pt"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PEOPLE_THRESHOLD = float(os.getenv("PEOPLE_THRESHOLD", "0.6"))  # below this -> "Unknown"
IOU = float(os.getenv("IOU", "0.4"))  # lower = fewer overlapping boxes
IMGSZ = int(os.getenv("IMGSZ", "960"))  # same size as in training
MAX_IMAGE_MB = 10
MAX_VIDEO_MB = 100
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "1800"))  # 1800 = about 60 s at 30 FPS
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Train first and put best.pt in /weights."
    )
model = YOLO(str(MODEL_PATH))
# Optional second model: tells whose signature it is. Without it, the app only detects.
people_model = YOLO(str(PEOPLE_PATH)) if PEOPLE_PATH.exists() else None

app = FastAPI(title="Signature Detection and Identification API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def to_detections(result):
    names = result.names
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "class_id": cls_id,
            "class_name": names[cls_id],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [x1, y1, x2, y2],
            "person": None,
            "person_confidence": None,
            "candidates": [],
        })
    return detections


def analyze(img, conf):
    """Detect signatures, identify the signer (if the people model exists), draw labels."""
    result = model.predict(img, conf=conf, imgsz=IMGSZ, iou=IOU, verbose=False)[0]
    detections = to_detections(result)
    if people_model is not None:
        for d in detections:
            crop = crop_box(img, d["bbox"])
            if crop is None:
                continue
            name, person_conf, candidates = identify(people_model, crop, PEOPLE_THRESHOLD)
            d["person"], d["person_confidence"], d["candidates"] = name, person_conf, candidates
    return detections, draw_detections(img, detections)


def check_conf(conf: float):
    if not 0.01 <= conf <= 0.99:
        raise HTTPException(400, "conf must be between 0.01 and 0.99")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": list(model.names.values()),
        "identification": people_model is not None,
        "people": [n.replace("_", " ") for n in people_model.names.values()] if people_model else [],
    }


@app.post("/predict/image")
def predict_image(file: UploadFile = File(...), conf: float = Form(0.25)):
    check_conf(conf)
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(400, "Please upload a JPG, PNG, WEBP or BMP image.")
    data = file.file.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_IMAGE_MB} MB.")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read this image.")

    t0 = time.perf_counter()
    detections, annotated = analyze(img, conf)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(500, "Could not encode the result image.")

    return {
        "image_base64": base64.b64encode(buf).decode(),
        "detections": detections,
        "count": len(detections),
        "identification_enabled": people_model is not None,
        "inference_ms": elapsed_ms,
        "width": img.shape[1],
        "height": img.shape[0],
    }


def make_browser_playable(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 with ffmpeg so every browser can play it."""
    if shutil.which("ffmpeg") is None:
        shutil.copy(src, dst)
        return False
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    done = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if done.returncode != 0:
        shutil.copy(src, dst)
        return False
    return True


@app.post("/predict/video")
def predict_video(file: UploadFile = File(...), conf: float = Form(0.25)):
    check_conf(conf)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, "Please upload an MP4, AVI, MOV or MKV video.")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{ext}"
        size = 0
        with open(src, "wb") as f:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_MB * 1024 * 1024:
                    raise HTTPException(413, f"Video is larger than {MAX_VIDEO_MB} MB.")
                f.write(chunk)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise HTTPException(400, "Could not read this video.")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frames, total, counts = 0, 0, {}
        t0 = time.perf_counter()
        while frames < MAX_VIDEO_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            detections, annotated = analyze(frame, conf)
            writer.write(annotated)
            for d in detections:
                key = d["person"] or d["class_name"]
                counts[key] = counts.get(key, 0) + 1
                total += 1
            frames += 1
        cap.release()
        writer.release()
        elapsed = time.perf_counter() - t0

        if frames == 0:
            raise HTTPException(400, "The video has no readable frames.")

        name = f"{uuid.uuid4().hex}.mp4"
        playable = make_browser_playable(raw, OUTPUT_DIR / name)

    return {
        "video_url": f"/outputs/{name}",
        "frames": frames,
        "total_detections": total,
        "class_counts": counts,
        "identification_enabled": people_model is not None,
        "processing_fps": round(frames / max(elapsed, 1e-6), 1),
        "browser_playable": playable,
        "truncated": frames >= MAX_VIDEO_FRAMES,
    }


# Serve the web frontend from the same server (must stay LAST so API routes win).
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
