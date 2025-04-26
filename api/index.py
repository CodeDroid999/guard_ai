from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# ------------------ MODEL LOADING ------------------

# Load your custom YOLO model
# ⚡ Replace 'yolo11n.pt' with your actual model path
model = YOLO('../yolo11n.pt')

# ----------------------------------------------------

# Utility to read image from bytes


def read_imagefile(file) -> np.ndarray:
    image = np.array(Image.open(io.BytesIO(file)))
    return image

# Utility to convert OpenCV image to bytes


def encode_image_to_bytes(img):
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# ------------------ API ROUTES ------------------


@app.route('/predict/single', methods=['POST'])
def predict_single_image():
    file = request.files['file']
    img = read_imagefile(file.read())

    results = model.predict(source=img, imgsz=640, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
    labels = results[0].boxes.cls.cpu().numpy().tolist()

    return jsonify({
        "boxes": boxes,
        "labels": labels,
    })


@app.route('/predict/batch', methods=['POST'])
def predict_batch_images():
    files = request.files.getlist('files')
    predictions = []

    for file in files:
        img = read_imagefile(file.read())
        results = model.predict(source=img, imgsz=640, conf=0.25)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        labels = results[0].boxes.cls.cpu().numpy().tolist()

        predictions.append({
            "filename": file.filename,
            "boxes": boxes,
            "labels": labels,
        })

    return jsonify(predictions)


@app.route('/predict/video', methods=['POST'])
def predict_video():
    file = request.files['file']
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)

    # Convert file into OpenCV VideoCapture
    video_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
    video = cv2.imdecode(video_bytes, cv2.IMREAD_COLOR)

    # Process frame-by-frame
    results = model.predict(source=video, imgsz=640, conf=0.25)

    frames = []
    for result in results:
        img = result.plot()  # Plot boxes
        img_bytes = encode_image_to_bytes(img)
        frames.append(img_bytes)

    return jsonify({"message": f"Processed {len(frames)} frames"})


@app.route('/predict/livestream', methods=['POST'])
def predict_live_stream():
    file = request.files['file']
    img = read_imagefile(file.read())

    results = model.predict(source=img, imgsz=640, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
    labels = results[0].boxes.cls.cpu().numpy().tolist()

    return jsonify({
        "boxes": boxes,
        "labels": labels,
    })


@app.route('/predict/webcam', methods=['GET'])
def predict_webcam():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        return jsonify({"error": "Cannot access webcam"})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=640, conf=0.25)
        annotated_frame = results[0].plot()

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Serve frame via streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(predict_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def root():
    return "YOLOv8 FastAPI Server Running 🚀"

# ------------------ MAIN ------------------


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
