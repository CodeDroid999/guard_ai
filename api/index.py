import os
import uuid
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
from ultralytics import solutions
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load model
model_path = os.path.join('model', 'yolo11n.pt')
securityalarm = solutions.SecurityAlarm(
    show=False, model=model_path, records=1)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400

    files = request.files.getlist('file')
    processed_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                processed_name = process_image(save_path)
            else:
                processed_name = process_video(save_path)

            processed_files.append(processed_name)

    return render_template('result.html', videos=processed_files)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                 cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        results = securityalarm(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    return output_filename


def process_image(image_path):
    im = cv2.imread(image_path)
    results = securityalarm(im)
    output_filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    cv2.imwrite(output_path, results.plot_im)
    return output_filename


@app.route('/stream/webcam')
def stream_webcam():
    return Response(generate_stream(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/cctv')
def stream_cctv():
    url = request.args.get('url')
    if not url:
        return "CCTV URL missing", 400
    return Response(generate_stream(url), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_stream(source):
    cap = cv2.VideoCapture(source)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = securityalarm(frame)
        ret, buffer = cv2.imencode('.jpg', results.plot_im)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
