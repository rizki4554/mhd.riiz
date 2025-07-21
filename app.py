from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)

model = YOLO('palingbaru.pt')

# Label map (dipotong untuk singkat)
label_map = {
    "1": "Aipysurus laevis (Berbisa)",
    "2": "Atractus trilineatus (Tidak Berbisa)",
    "3": "Boiga cyanea (Berbisa)",
    # ...
    "59": "Xenochrophis trianguligerus (Tidak Berbisa)"
}

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_colors(num_classes):
    np.random.seed(42)
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

CLASS_COLORS = generate_colors(len(label_map))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/deteksi')
def index():
    return render_template('webcam_live.html')  # Masih ditampilkan, tapi nonaktif

@app.route('/video_feed')
def video_feed():
    return "Fitur webcam tidak tersedia di versi hosting Railway."

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            results = model.predict(source=filepath, conf=0.4, save=False, verbose=False)
            result = results[0]
            result_path = os.path.join(UPLOAD_FOLDER, 'result_' + filename)
            result.save(filename=result_path)

            if result.boxes:
                kelas = int(result.boxes.cls[0])
                label = model.names[kelas]
                confidence = float(result.boxes.conf[0]) * 100
                label_text = label_map.get(str(label), f"Label {label}")
            else:
                label_text = 'Tidak ada ular terdeteksi'
                confidence = 0.0

            image_path = url_for('static', filename='uploads/' + os.path.basename(result_path))

            return render_template('upload_result.html',
                                   image_path=image_path,
                                   label=label_text,
                                   confidence=round(confidence, 2))
    return render_template('upload_page.html')

# Bagian penting untuk Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
