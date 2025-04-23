from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
import uuid
from ultralytics import solutions

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Email config
from_email = "abc@gmail.com"
password = "your_16_digit_app_password"
to_email = "xyz@gmail.com"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    video = request.files['video']
    if not video:
        return redirect(url_for('index'))

    # Save the uploaded video
    filename = f"{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(video_path)

    # Initialize capture and security alarm
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    output_filename = f"processed_{filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    securityalarm = solutions.SecurityAlarm(
        show=False,
        model="model/yolo11n.pt",
        records=1,
    )
    securityalarm.authenticate(from_email, password, to_email)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        results = securityalarm(im0)
        writer.write(results.plot_im)

    cap.release()
    writer.release()
    return render_template('index.html', video_path=output_path)

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)
