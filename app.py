from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load model (pastikan model .pt sudah diunduh di Railway atau gunakan URL publik)
model = YOLO('palingbaru.pt')

# Mapping label dari YOLO ke nama ular
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/deteksi')
def index():
    return render_template('webcam_live.html')  # Versi JavaScript

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

            # Prediksi
            results = model.predict(source=filepath, conf=0.4, save=False, verbose=False)
            result = results[0]
            result_path = os.path.join(UPLOAD_FOLDER, 'result_' + filename)
            result.save(filename=result_path)

            # Ambil hasil
            if result.boxes:
                kelas = int(result.boxes.cls[0])
                label = model.names[kelas]
                confidence = float(result.boxes.conf[0]) * 100
                label_text = label_map.get(str(label), f"Label {label}")
            else:
                label_text = 'Tidak ada ular terdeteksi'
                confidence = 0.0

            image_path = url_for('static', filename='uploads/' + os.path.basename(result_path))

            return f"""
            <h3>📷 Hasil Deteksi</h3>
            <img src="{image_path}" width="640" />
            <p><strong>{label_text}</strong><br>Confidence: {round(confidence, 2)}%</p>
            <a href="/deteksi">🔁 Coba Lagi</a> | <a href="/">🏠 Kembali</a>
            """
    return render_template('upload_page.html')

if __name__ == '__main__':
    app.run(debug=True)
