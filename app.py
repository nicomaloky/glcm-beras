import os
import cv2
import numpy as np
import pandas as pd
import joblib
import base64
import re
import io
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template, jsonify


#Konfigurasi Path
DATASET_PATH = 'dataset/'
MODEL_PATH = 'model_rf.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'

#Daftar nama untuk semua fitur
TEXTURE_FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
COLOR_FEATURES = ['hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std']
ALL_FEATURES = TEXTURE_FEATURES + COLOR_FEATURES


#Key di sini (huruf kecil) harus sama persis dengan nama folder di dataset.
DESKRIPSI_BERAS = {
    "arborio": "Beras Arborio berasal dari Italia, memiliki bulir pendek dan kandungan pati tinggi. Sangat cocok untuk membuat risotto karena menghasilkan tekstur yang creamy.",
    "basmati": "Dikenal dengan aroma wanginya yang khas dan bulir yang panjang. Saat dimasak, nasi Basmati tidak lengket dan tetap terpisah. Ideal untuk biryani dan pilaf.",
    "ipsala": "Beras jenis ini populer di Turki, memiliki bulir sedang dan tekstur yang pulen setelah dimasak. Sering digunakan untuk hidangan nasi sehari-hari.",
    "jasmine": "Beras Jasmine dari Thailand memiliki aroma wangi pandan yang lembut. Teksturnya sedikit lengket dan lembut, cocok untuk hidangan Asia Tenggara.",
    "karacadag": "Beras unik dari daerah Karacadağ, Turki. Memiliki bulir yang lebih gelap dan rasa yang khas. Tahan terhadap kondisi kering saat ditanam.",
    "default": "Deskripsi untuk jenis beras ini belum tersedia."
}


def ekstrak_fitur_tekstur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return [np.mean(graycoprops(glcm, prop)) for prop in TEXTURE_FEATURES]

def ekstrak_fitur_warna(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
    return [
        np.mean(h[mask]), np.std(h[mask]),
        np.mean(s[mask]), np.std(s[mask]),
        np.mean(v[mask]), np.std(v[mask])
    ]

def ekstrak_fitur_gabungan(path_gambar):
    try:
        img = cv2.imread(path_gambar)
        if img is None: return None
        return ekstrak_fitur_tekstur(img) + ekstrak_fitur_warna(img)
    except Exception as e:
        print(f"Error memproses gambar {path_gambar}: {e}")
        return None

def latih_dan_simpan_model():
    print("Memulai proses ekstraksi fitur gabungan (Tekstur + Warna)...")
    fitur_list, label_list = [], []

    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        raise FileNotFoundError(f"Folder 'dataset' tidak ditemukan atau kosong.")

    for nama_kelas in sorted(os.listdir(DATASET_PATH)):
        path_kelas = os.path.join(DATASET_PATH, nama_kelas)
        if os.path.isdir(path_kelas):
            print(f"  Memproses kelas: {nama_kelas}")
            for nama_file in os.listdir(path_kelas):
                path_file = os.path.join(path_kelas, nama_file)
                fitur = ekstrak_fitur_gabungan(path_file)
                if fitur:
                    fitur_list.append(fitur)
                    label_list.append(nama_kelas)

    if not fitur_list: raise ValueError("Tidak ada fitur yang berhasil diekstrak.")

    df = pd.DataFrame(fitur_list, columns=ALL_FEATURES)
    df['label'] = label_list
    print("Ekstraksi fitur selesai.")

    X, y = df.drop('label', axis=1), df['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Memulai pelatihan model Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    akurasi = accuracy_score(y_test, y_pred)
    print(f"Pelatihan Selesai. Akurasi Model Random Forest: {akurasi * 100:.2f}%")
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model, scaler, dan encoder baru telah disimpan.")
    return model, scaler, le

#Inisialiasi aplikasi flask dan model

app = Flask(__name__)

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
    print("File model tidak ditemukan. Memulai proses pelatihan otomatis...")
    try:
        model, scaler, le = latih_dan_simpan_model()
    except (FileNotFoundError, ValueError) as e:
        print(f"FATAL ERROR saat pelatihan: {e}")
        exit()
else:
    print("Memuat model Random Forest yang sudah ada...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    print("Model berhasil dimuat.")


#Routes dan fungsi prediksi

def base64_to_cv2_image(base64_string):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_string)
    img_bytes = base64.b64decode(base64_data)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Data gambar tidak ditemukan"}), 400

    try:
        img_cv2 = base64_to_cv2_image(data['image'])
        
        fitur_raw = ekstrak_fitur_tekstur(img_cv2) + ekstrak_fitur_warna(img_cv2)
        fitur_gambar = np.array(fitur_raw).reshape(1, -1)
        
        fitur_scaled = scaler.transform(fitur_gambar)
        prediksi_encoded = model.predict(fitur_scaled)
        prediksi_proba = model.predict_proba(fitur_scaled)
        
        nama_jenis_beras = le.inverse_transform(prediksi_encoded)[0]
        akurasi_prediksi = np.max(prediksi_proba) * 100
        
        fitur_dict = {ALL_FEATURES[i]: f"{fitur_raw[i]:.4f}" for i in range(len(ALL_FEATURES))}
        nama_jenis_beras_key = nama_jenis_beras.lower().replace(' ', '_')
        deskripsi = DESKRIPSI_BERAS.get(nama_jenis_beras_key, DESKRIPSI_BERAS["default"])
        
        hasil = {
            "jenis_beras": nama_jenis_beras.capitalize().replace('_', ' '),
            "akurasi": f"{akurasi_prediksi:.2f}",
            "fitur": fitur_dict,
            "deskripsi": deskripsi
        }
        return jsonify(hasil)
        
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({"error": f"Gagal melakukan prediksi: {str(e)}"}), 500

#Menjalankan Aplikasi

if __name__ == '__main__':
    app.run(debug=True)
