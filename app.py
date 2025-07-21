from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Lazy loading untuk YOLO - akan di-import saat diperlukan
model = None

def get_model():
    """Lazy load YOLO model untuk menghindari error saat startup"""
    global model
    if model is None:
        try:
            from ultralytics import YOLO
            model = YOLO('palingbaru.pt')
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None
    return model

# Label map ular dengan mapping yang lebih clean
LABEL_MAP = {
    "0": "Acrochordus granulatus (Tidak Berbisa)",
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

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads'
CONFIDENCE_THRESHOLD = 0.4
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Cek apakah file yang diupload diizinkan"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_colors(num_classes):
    """Generate warna unik per kelas dengan seed tetap"""
    colors = []
    np.random.seed(42)
    for i in range(num_classes):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors.append(color)
    return colors

# Generate warna untuk semua kelas
CLASS_COLORS = generate_colors(len(LABEL_MAP))

@app.route('/')
def home():
    """Halaman utama"""
    return render_template('home.html')

@app.route('/deteksi')
def deteksi():
    """Halaman deteksi real-time via webcam"""
    return render_template('webcam_live.html')

@app.route('/health')
def health_check():
    """Health check endpoint untuk deployment"""
    model_status = get_model() is not None
    return jsonify({
        'status': 'healthy' if model_status else 'unhealthy',
        'model_loaded': model_status
    })

def gen_frames():
    """Generator untuk streaming video dari webcam"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Get model instance
            current_model = get_model()
            if current_model is None:
                # Jika model belum loaded, tampilkan frame kosong dengan pesan
                cv2.putText(frame, "Model loading...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                try:
                    # Prediksi menggunakan YOLO
                    results = current_model.predict(
                        source=frame, 
                        conf=CONFIDENCE_THRESHOLD, 
                        save=False, 
                        verbose=False
                    )
                    
                    result = results[0]
                    annotated_frame = frame.copy()

                    # Gambar bounding box jika ada deteksi
                    if result.boxes:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # Ambil label dari model YOLO
                            label = current_model.names[cls_id]
                            label_text = LABEL_MAP.get(str(label), f"Unknown ({label})")
                            
                            # Koordinat bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Warna berdasarkan kelas
                            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

                            # Gambar kotak deteksi
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                            # Text label dan confidence
                            text = f"{label_text} ({conf * 100:.1f}%)"
                            
                            # Background untuk text
                            (text_width, text_height), baseline = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                            )
                            cv2.rectangle(
                                annotated_frame, 
                                (x1, y1 - text_height - 10), 
                                (x1 + text_width, y1), 
                                color, -1
                            )
                            
                            cv2.putText(
                                annotated_frame, text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                            )

                    frame = annotated_frame
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    cv2.putText(frame, "Prediction error", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    """Stream video untuk webcam real-time"""
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle upload dan prediksi gambar"""
    if request.method == 'POST':
        # Validasi file
        if 'image' not in request.files:
            return render_template('upload_page.html', error='Tidak ada file yang dipilih')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('upload_page.html', error='Tidak ada file yang dipilih')
        
        if not allowed_file(file.filename):
            return render_template('upload_page.html', 
                                 error='Format file tidak didukung. Gunakan: PNG, JPG, JPEG, GIF, BMP')
        
        # Check file size
        if hasattr(file, 'content_length') and file.content_length > MAX_FILE_SIZE:
            return render_template('upload_page.html', 
                                 error='File terlalu besar. Maksimal 16MB')
        
        try:
            # Simpan file dengan timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
            file.save(filepath)

            # Get model dan lakukan prediksi
            current_model = get_model()
            if current_model is None:
                return render_template('upload_page.html', 
                                     error='Model belum siap. Coba lagi dalam beberapa saat.')

            # Prediksi menggunakan YOLO
            results = current_model.predict(
                source=filepath, 
                conf=CONFIDENCE_THRESHOLD, 
                save=False, 
                verbose=False
            )
            
            result = results[0]
            
            # Buat gambar hasil dengan annotasi
            annotated_img = result.plot()
            result_filename = f'result_{safe_filename}'
            result_path = os.path.join(UPLOAD_FOLDER, result_filename)
            
            # Simpan gambar hasil
            cv2.imwrite(result_path, annotated_img)

            # Ekstrak hasil prediksi
            detections = []
            if result.boxes:
                for box in result.boxes:
                    kelas = int(box.cls[0])
                    label = current_model.names[kelas]
                    confidence = float(box.conf[0]) * 100
                    label_text = LABEL_MAP.get(str(label), f"Unknown ({label})")
                    
                    detections.append({
                        'label': label_text,
                        'confidence': round(confidence, 2),
                        'venomous': 'Berbisa' in label_text
                    })
            
            # Jika tidak ada deteksi
            if not detections:
                detections = [{
                    'label': 'Tidak ada ular terdeteksi',
                    'confidence': 0.0,
                    'venomous': False
                }]

            image_url = url_for('static', filename=f'uploads/{result_filename}')

            return render_template('upload_result.html',
                                 image_path=image_url,
                                 detections=detections,
                                 total_detections=len([d for d in detections if d['confidence'] > 0]))

        except Exception as e:
            print(f"Error during upload processing: {e}")
            # Cleanup file jika ada error
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('upload_page.html', 
                                 error=f'Terjadi kesalahan saat memproses gambar: {str(e)}')
    
    return render_template('upload_page.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
