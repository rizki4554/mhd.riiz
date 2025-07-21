from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np
import urllib.request
import logging

# Logging debug saat awal runtime untuk Railway
print("[DEBUG] App starting...")

app = Flask(__name__)

# Konfigurasi logging lebih eksplisit
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("app")
logger.debug("Initializing app...")

# Konstanta untuk model
MODEL_PATH = "palingbaru.pt"
GDRIVE_ID = "1rOva-D9aOBxApVI_YKcKSxvPLw5xHs0U"

# Unduh model jika belum ada
if not os.path.exists(MODEL_PATH):
    logger.info("[DEBUG] Model belum ada, mulai download dari Google Drive...")
    gdown_url = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"
    urllib.request.urlretrieve(gdown_url, MODEL_PATH)
    logger.info("[DEBUG] Download model selesai!")
else:
    logger.info("[DEBUG] Model sudah tersedia, tidak perlu download.")

# Load model
model = YOLO(MODEL_PATH)
logger.info("[DEBUG] Model YOLO berhasil diload.")

# Label map (dipersingkat agar tidak panjang di sini)
label_map = {
     "31 0 0 0 1 1 1 1 0 0 0": "Acrochordus granulatus (Tidak Berbisa)",
    "1": "Aipysurus laevis (Berbisa)",
    "2": "Atractus trilineatus (Tidak Berbisa)",
    "3": "Boiga cyanea (Berbisa)",
    "4": "Boiga cynodon (Berbisa)",
    "5": "Boiga kraepelini (Tidak Berbisa)",
    "6": "Bothrops ammodytoides (Berbisa)",
    "7": "Bothrops bilineatus (Berbisa)",
    "8": "Bungarus candidus (Berbisa)",
    "9": "Cacophis harriettae (Tidak Berbisa)",
    "10": "Chironius monticola (Tidak Berbisa)",
    "11": "Clelia equatoriana (Tidak Berbisa)",
    "12": "Corallus annulatus (Tidak Berbisa)",
    "13": "Crotalus mitchellii (Berbisa)",
    "14": "Cubophis cantherigerus (Berbisa)",
    "15": "Drymoluber dichrous (Tidak Berbisa)",
    "16": "Echis carinatus (Berbisa)",
    "17": "Elaphe anomala (Tidak Berbisa)",
    "18": "Erythrolamprus melanotus (Tidak Berbisa)",
    "19": "Erythrolamprus typhlus (Tidak Berbisa)",
    "20": "Furina ornata (Tidak Berbisa)",
    "21": "Geophis sartorii (Tidak Berbisa)",
    "22": "Heterodon simus (Tidak Berbisa)",
    "23": "Hoplocephalus stephensii (Berbisa)",
    "24": "Hypnale hypnale (Berbisa)",
    "25": "Hypsirhynchus parvifrons (Tidak Berbisa)",
    "26": "Imantodes gemmistratus (Tidak Berbisa)",
    "27": "Lampropeltis annulata (Tidak Berbisa)",
    "28": "Leptodeira splendida (Tidak Berbisa)",
    "29": "Loxocemus bicolor (Tidak Berbisa)",
    "30": "Lygophis lineatus (Tidak Berbisa)",
    "31": "Macrelaps microlepidotus (Berbisa)",
    "32": "Macroprotodon mauritanicus (Berbisa)",
    "33": "Macrovipera lebetinus (Berbisa)",
    "34": "Masticophis bilineatus (Tidak Berbisa)",
    "35": "Masticophis fuliginosus (Tidak Berbisa)",
    "36": "Micruroides euryxanthus (Berbisa)",
    "37": "Micrurus browni (Berbisa)",
    "38": "Micrurus distans (Berbisa)",
    "39": "Micrurus mosquitensis (Berbisa)",
    "40": "Micrurus pyrrhocryptus (Berbisa)",
    "41": "Micrurus surinamensis (Berbisa)",
    "42": "Naja annulifera (Berbisa)",
    "43": "Naja mossambica (Berbisa)",
    "44": "Naja naja (Berbisa)",
    "45": "Naja siamensis (Berbisa)",
    "46": "Pantherophis alleghaniensis (Tidak Berbisa)",
    "47": "Pantherophis guttatus (Tidak Berbisa)",
    "48": "Paraphimophis rusticus (Tidak Berbisa)",
    "49": "Pituophis melanoleucus (Tidak Berbisa)",
    "50": "Porthidium lansbergii (Berbisa)",
    "51": "Pseudonaja mengdeni (Berbisa)",
    "52": "Ptyas fusca (Tidak Berbisa)",
    "53": "Rhabdophis plumbicolor (Berbisa)",
    "54": "Rhabdophis siamensis (Berbisa)",
    "55": "Salvadora grahamiae (Tidak Berbisa)",
    "56": "Salvadora mexicana (Tidak Berbisa)",
    "57": "Thelotornis capensis (Berbisa)",
    "58": "Trimeresurus popeiorum (Berbisa)",
    "59": "Xenochrophis trianguligerus (Tidak Berbisa)"
}

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Warna per kelas
np.random.seed(42)
CLASS_COLORS = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(label_map))]

@app.route('/')
def home():
    logger.debug("[DEBUG] Home route diakses.")
    return render_template('home.html')

@app.route('/deteksi')
def index():
    logger.debug("[DEBUG] Halaman deteksi (webcam) dibuka.")
    return render_template('webcam_live.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("[WARNING] Gagal membaca frame dari kamera.")
            break
        results = model.predict(source=frame, conf=0.4, save=False, verbose=False)
        result = results[0]
        annotated_frame = frame.copy()

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            label_text = label_map.get(str(label), f"Label {label}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label_text} ({conf * 100:.1f}%)"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
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

        return render_template('upload_result.html', image_path=image_path, label=label_text, confidence=round(confidence, 2))
    return render_template('upload_page.html')

if __name__ == '__main__':
    app.run(debug=True)
