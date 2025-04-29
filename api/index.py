from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import cv2
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model
model_path = "../yolo11n.pt"  # <-- Specify the correct model path
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
    """Render the home page."""
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload for image/video processing."""
    if "file" not in request.files:
        return "No file part", 400

    files = request.files.getlist("file")
    results = []

    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        if file.filename.lower().endswith((".mp4", ".avi", ".mov")):
            # It's a video, no prediction required
            results.append(f"Video uploaded: {file.filename}")
        else:
            # It's an image, run YOLO prediction
            pred = model.predict(source=filepath, save=True,
                                 project="runs", name="predictions", exist_ok=True)
            results.append(f"Image predicted: {file.filename}")

    return redirect(url_for('home'))


@app.route("/stream/webcam")
def stream_webcam():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    return render_template("stream.html")


@app.route("/stream/cctv")
def stream_cctv():
    """Stream video from a provided CCTV URL."""
    global video_capture, stream_url
    stream_url = request.args.get("url")
    if not stream_url:
        return "No CCTV URL provided", 400

    video_capture = cv2.VideoCapture(stream_url)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video")
def video():
    """Stream video."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
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

        # YOLO Prediction
        results = model.predict(frame, imgsz=640, conf=0.5)
        annotated_frame = results[0].plot()

        # Extract object labels and confidences
        detections = []
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0]) * 100
            detections.append(f"{cls}: {conf:.2f}% confidence")

        # Simulate action inference (you can integrate your real model)
        predicted_action = "Stealing"
        predicted_conf = 40.0
        all_scores = {
            "Stealing": 40.0,
            "Sneaking": 0.0,
            "Peaking": 0.0,
            "Normal": 0.0
        }

        # Format overlay text
        info_text = "Detected Objects:\n" + \
            "\n".join(f"- {d}" for d in detections)
        info_text += f"\n\nAction Analysis:\nPredicted Action: {predicted_action} ({predicted_conf:.2f}% confidence)"
        info_text += "\n\nAll Action Scores:\n" + \
            "\n".join(f"- {k}: {v:.2f}%" for k, v in all_scores.items())

        # Add overlay text to the frame
        y0, dy = 30, 25
        for i, line in enumerate(info_text.split("\n")):
            y = y0 + i * dy
            cv2.putText(annotated_frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ================== MAIN ====================

if __name__ == "__main__":
    # Run the Flask app on the dynamic port provided by the environment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
