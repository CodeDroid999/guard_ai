from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import cv2
import torch
from ultralytics import YOLO
import os

app = Flask(
    __name__,
    static_url_path='../static',      # Allow /static access
    static_folder='static',          # Where your CSS/JS/Images live
    template_folder='templates'      # Where your HTML files live
)

# Load YOLO model
model_path = "../yolo11n.pt"  # <-- your model path
model = YOLO(model_path)

# Global variables for video streaming
video_capture = None
stream_url = None

# Ensure uploads directory exists
UPLOAD_FOLDER = '../uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== ROUTES ====================


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400

    files = request.files.getlist("file")
    results = []

    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        if file.filename.lower().endswith((".mp4", ".avi", ".mov")):
            # It's a video
            results.append(f"Video uploaded: {file.filename}")
        else:
            # It's an image -> run YOLO prediction
            pred = model.predict(source=filepath, save=True,
                                 project="runs", name="predictions", exist_ok=True)
            results.append(f"Image predicted: {file.filename}")

    return redirect(url_for('home'))


@app.route("/stream/webcam")
def stream_webcam():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream/cctv")
def stream_cctv():
    global video_capture, stream_url
    stream_url = request.args.get("url")
    if not stream_url:
        return "No CCTV URL provided", 400

    video_capture = cv2.VideoCapture(stream_url)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Serve uploaded/processed files (if needed)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================== STREAM FUNCTION ====================


def generate_frames():
    global video_capture
    while True:
        if video_capture is None:
            break
        success, frame = video_capture.read()
        if not success:
            break

        # Run YOLO prediction on frame
        results = model.predict(frame, imgsz=640, conf=0.5)
        frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ================== MAIN ====================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
