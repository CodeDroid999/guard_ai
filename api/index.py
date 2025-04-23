# app.py
import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load model from model directory
model_path = os.path.join('model', 'yolo11n.pt')
model = YOLO(model_path)

# Capture from webcam
cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, show=False)

        # Plot detection results on frame
        for r in results:
            annotated_frame = r.plot()

        # Encode the frame for MJPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')  # HTML to show video stream


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
