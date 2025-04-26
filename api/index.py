from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load your custom YOLO model here
model_path = "../yolo11n.pt"  # <--- leave this for your own model
model = YOLO(model_path)

# Global variables for stream sources
video_capture = None
stream_url = None


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400
    files = request.files.getlist("file")
    os.makedirs("uploads", exist_ok=True)

    results = []

    for file in files:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        if file.filename.lower().endswith((".mp4", ".avi", ".mov")):
            # Handle video file
            results.append(f"Video uploaded: {file.filename}")
        else:
            # Handle image prediction
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


def generate_frames():
    global video_capture
    while True:
        if video_capture is None:
            break
        success, frame = video_capture.read()
        if not success:
            break

        # Inference on each frame
        results = model.predict(frame, imgsz=640, conf=0.5)
        frame = results[0].plot()  # Draw predictions

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
