from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load your model (replace with the actual path to your model)
model = torch.load('yolo11n.pt')
model.eval()

# Image transformation (adjust based on model requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Helper function for running inference on an image


def predict_image(image: Image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    return output.tolist()

# Endpoint for uploading single images for prediction


@app.route('/upload-image', methods=['POST'])
def upload_image():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    prediction = predict_image(image)
    return jsonify({'prediction': prediction})

# Endpoint for uploading a batch of images for prediction


@app.route('/upload-batch', methods=['POST'])
def upload_batch():
    files = request.files.getlist('files')
    predictions = []
    for file in files:
        image = Image.open(io.BytesIO(file.read()))
        prediction = predict_image(image)
        predictions.append({'prediction': prediction})
    return jsonify(predictions)

# Endpoint for handling webcam stream frames (live stream)


@app.route('/live-stream', methods=['POST'])
def live_stream():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    prediction = predict_image(image)
    return jsonify({'prediction': prediction})

# Endpoint for uploading video files for frame-by-frame prediction


@app.route('/upload-video', methods=['POST'])
def upload_video():
    file = request.files['file']
    video_data = file.read()
    video_file = io.BytesIO(video_data)

    # Open the video file using OpenCV
    video = cv2.VideoCapture(video_file)
    predictions = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Process each frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prediction = predict_image(image)
        predictions.append({'prediction': prediction})

    video.release()
    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    # Run Flask app on all available network interfaces and a specific port
    app.run(debug=True, host='0.0.0.0', port=5000)
