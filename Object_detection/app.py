from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
import urllib.request

if not os.path.exists("yolov8m.pt"):
    print("🔄 Downloading YOLOv8 model...")
    url = "https://drive.google.com/uc?export=download&id=1twpYVmRJw8OhpPOXdPu13JPl8zUAyJE8"
    urllib.request.urlretrieve(url, "yolov8m.pt")
    print("✅ Download complete.")
else:
    print("✅ Model already exists. Skipping download.")


model = YOLO("yolov8m.pt")  
tracker = DeepSort(max_age=30)

def generate_frames(video_source):
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

        # Deep SORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            label = track.get_det_class() if hasattr(track, 'get_det_class') else 'Object'
            cv2.putText(frame, f'{label} ID:{track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', stream=None)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return render_template('index.html', stream=url_for('video_feed', path=filename))
    return redirect(url_for('index'))

@app.route('/webcam', methods=['POST'])
def webcam():
    return render_template('index.html', stream=url_for('video_feed'))

@app.route('/video_feed')
@app.route('/video_feed/<path>')
def video_feed(path=None):
    source = 0 if not path else os.path.join(UPLOAD_FOLDER, path)
    return Response(generate_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
