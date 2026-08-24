import streamlit as st
import keras
from PIL import Image
import numpy as np
import os
import zipfile
import json
import tempfile
import shutil

# --- BACA FILE .env SECARA MANUAL JIKA ADA ---
def load_env_file():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip().strip("'\"").strip()
                    os.environ[key.strip()] = val

load_env_file()

# --- AMBIL GEMINI API KEY DARI ENVIRONMENT / STREAMLIT SECRETS ---
def get_api_key():
    load_env_file()
    key = os.environ.get("GEMINI_API_KEY", "").strip().strip("'\"")
    if key:
        return key
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"].strip().strip("'\"")
    except Exception:
        pass
    return ""

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AquaDoc AI - Smart Fish Disease Classifier",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN GLASSMORPHISM & VIBRANT UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner Gradient */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #0891b2 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px -5px rgba(2, 132, 199, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        color: white;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #E0F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #BAE6FD;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Custom Glass Cards */
    .custom-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .custom-card:hover {
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.08);
    }

    /* Result Banner */
    .diagnosis-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 20px -3px rgba(3, 105, 161, 0.3);
    }

    .diagnosis-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        font-weight: 600;
    }

    .diagnosis-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 0.5rem 1.1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .confidence-val {
        font-size: 1.4rem;
        font-weight: 800;
    }

    /* LLM Medical Report Box */
    .llm-report {
        background: linear-gradient(180deg, #faf5ff 0%, #f5f3ff 100%);
        border: 1px solid #e9d5ff;
        border-left: 5px solid #9333ea;
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(147, 51, 234, 0.08);
        color: #1e1b4b;
        line-height: 1.7;
    }

    .llm-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #9333ea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 6px 15px -2px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px -2px rgba(2, 132, 199, 0.5) !important;
    }

    /* Status indicator */
    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-online {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .status-offline {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    </style>
""", unsafe_allow_html=True)

# --- KONSTANTA PATH MODEL & DAFTAR KELAS ---
MODEL_PATH = "model.keras"

CLASS_NAMES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
]

# --- FUNGSI LOAD MODEL DENGAN CACHE & CONFIG SANITIZER ---
def _sanitize_keras_config(obj):
    if isinstance(obj, dict):
        obj.pop('quantization_config', None)
        for k, v in list(obj.items()):
            obj[k] = _sanitize_keras_config(v)
    elif isinstance(obj, list):
        obj = [_sanitize_keras_config(item) for item in obj]
    return obj

@st.cache_resource
def load_fish_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception:
        try:
            temp_dir = tempfile.mkdtemp()
            temp_model_path = os.path.join(temp_dir, "cleaned_model.keras")
            with zipfile.ZipFile(model_path, "r") as zin, zipfile.ZipFile(temp_model_path, "w") as zout:
                for item in zin.infolist():
                    buffer = zin.read(item.filename)
                    if item.filename == "config.json":
                        cfg = json.loads(buffer.decode("utf-8"))
                        cfg = _sanitize_keras_config(cfg)
                        buffer = json.dumps(cfg).encode("utf-8")
                    zout.writestr(item, buffer)
            
            loaded_model = keras.models.load_model(temp_model_path, compile=False)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return loaded_model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None

# --- FUNGSI GENERATE DESKRIPSI, GEJALA & SARAN PENANGANAN VIA LLM ---
def generate_llm_analysis(api_key: str, disease_name: str, confidence: float, class_scores: dict, additional_notes: str = ""):
    api_key = api_key.strip().strip("'\"")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
Anda adalah seorang dokter hewan spesialis akuakultur dan patologi perikanan (Senior Fish Health Specialist).
Sebuah model Deep Learning Computer Vision baru saja mengklasifikasikan foto ikan dengan hasil:

- **Diagnosis Terdeteksi:** {disease_name}
- **Tingkat Keyakinan (Confidence):** {confidence:.2f}%
- **Distribusi Probabilitas Semua Kelas:**
{json.dumps(class_scores, indent=2)}
- **Catatan Tambahan Kondisi Ikan:** {additional_notes if additional_notes else "Tidak ada catatan khusus dari pembudidaya."}

Susunlah laporan diagnosis klinis dan panduan tindakan darurat secara mendalam dalam Bahasa Indonesia dengan format berikut:

### 1. 🔍 Ringkasan Diagnosis & Gejala Klinis
- Penjelasan patogen penyebab (`{disease_name}`: bakteri/virus/jamur/parasit).
- Ciri-ciri lesi fisik eksternal, kondisi insang, sisik, atau perubahan pola renang ikan.

### 2. ⚠️ Tingkat Keparahan & Resiko Penularan
- Derajat kegawatan (Rendah / Sedang / Tinggi / Kritis).
- Potensi mortalitas dan kecepatan penularan di ekosistem kolam/akuarium.

### 3. 💊 Protokol Tindakan Medis & Karantina (Step-by-Step)
- **Tindakan Darurat 24 Jam Pertama:** Isolasi, pergantian air, dan kontrol aerasi.
- **Rekomendasi Terapi & Dosis:** Obat/antiseptik/antibiotik/garam ikan dengan dosis aman (g/L atau ppm).

### 4. 🌊 Parameter Kualitas Air & Manajemen Lingkungan
- Standar kualitas air optimal (Suhu ideal, pH, Toleransi Amonia/Nitrit, dan DO).
- Biosekuriti agar wabah tidak terulang.

Gunakan format markdown yang rapi, informatif, dan profesional.
"""
        for model_id in ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt
                )
                return response.text
            except Exception as inner_err:
                last_err = inner_err
                continue
        raise last_err
    except Exception as e:
        return f"⚠️ Gagal menghubungi LLM: {e}"

# --- SIDEBAR KONFIGURASI ---
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi Sistem")
    
    normalize_option = st.selectbox(
        "Metode Preprocessing Gambar:",
        [
            "ResNet50 Standard (keras.applications.resnet50)",
            "Skala 0 - 1 (dibagi 255)", 
            "Skala -1 hingga 1", 
            "Tanpa Normalisasi (0 - 255)"
        ],
        index=0,
        help="Pilih standarisasi piksel yang cocok dengan tahapan pelatihan model"
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Integrasi Gemini LLM")
    gemini_api_key = get_api_key()
    
    if gemini_api_key:
        st.markdown('<div class="status-pill status-online">● API Key Terhubung (.env)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-offline">● API Key Tidak Ditemukan</div>', unsafe_allow_html=True)
        gemini_api_key = st.text_input(
            "Masukkan API Key Manual:",
            type="password",
            help="Tambahkan GEMINI_API_KEY di file .env"
        )
    
    st.markdown("---")
    st.markdown("### ℹ️ Tentang Model")
    st.caption("• **Arsitektur:** ResNet50 Transfer Learning")
    st.caption("• **Input Shape:** 256 × 256 × 3")
    st.caption("• **Output:** 7 Penyakit Ikan Air Tawar/Laut")
    st.caption("• **LLM Engine:** Gemini 3.6 Flash")

# Memuat model Keras
model = load_fish_model(MODEL_PATH)

# --- HERO BANNER HEADER ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🔬 DEEP LEARNING + LLM CLINICAL INSIGHT</div>
    <div class="hero-title">AquaDoc AI: Deteksi Penyakit Ikan</div>
    <div class="hero-subtitle">
        Sistem cerdas klasifikasi penyakit ikan berbasis Deep Learning Computer Vision yang diintegrasikan dengan asistensi medis otomatis dari LLM Gemini.
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ File model `{MODEL_PATH}` tidak ditemukan di direktori utama projek. Pastikan file model sudah tersedia.")

# --- DUA KOLOM LAYOUT UTAMA ---
col1, col2 = st.columns([1, 1.15], gap="large")

with col1:
    st.markdown("### 📤 Unggah Sampel Ikan")
    uploaded_file = st.file_uploader(
        "Upload foto ikan bergejala (Format: JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Pastikan foto memperlihatkan bagian tubuh ikan yang sakit dengan pencahayaan jelas"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Foto Sampel yang Dianalisis", use_container_width=True)
    
    user_notes = st.text_area(
        "📝 Catatan Tambahan (Kondisi Ikan & Air):",
        placeholder="Contoh: Ikan berenang miring dan megap-megap di permukaan, air kolam agak keruh dan berbusa...",
        help="Informasi ini akan dianalisis oleh Dokter Hewan LLM untuk rekomendasi penanganan yang lebih akurat."
    )

with col2:
    st.markdown("### 🩺 Hasil Diagnosis & Konsultasi")
    
    if uploaded_file is None:
        st.info("👈 Silakan unggah foto ikan di panel sebelah kiri untuk memulai diagnosis.")
    elif model is None:
        st.error(f"Model `{MODEL_PATH}` belum berhasil dimuat.")
    else:
        # Preprocessing Gambar
        target_size = (256, 256)
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(image_resized, dtype=np.float32)

        if normalize_option == "ResNet50 Standard (keras.applications.resnet50)":
            import keras.applications.resnet50 as resnet_prep
            img_processed = resnet_prep.preprocess_input(img_array.copy())
        elif normalize_option == "Skala 0 - 1 (dibagi 255)":
            img_processed = img_array / 255.0
        elif normalize_option == "Skala -1 hingga 1":
            img_processed = (img_array / 127.5) - 1.0
        else:
            img_processed = img_array

        input_data = np.expand_dims(img_processed, axis=0)

        if st.button("🚀 Jalankan Diagnosis & Analisis Medis", type="primary", use_container_width=True):
            with st.spinner("🔍 Menganalisis citra dengan Deep Learning Model..."):
                try:
                    preds = model.predict(input_data)
                    
                    if preds.shape[-1] == 1:
                        prob = float(preds[0][0])
                        pred_idx = 1 if prob >= 0.5 else 0
                        confidence = prob * 100 if pred_idx == 1 else (1.0 - prob) * 100
                    else:
                        pred_idx = int(np.argmax(preds[0]))
                        confidence = float(preds[0][pred_idx] * 100)

                    if pred_idx < len(CLASS_NAMES):
                        result_name = CLASS_NAMES[pred_idx]
                    else:
                        result_name = f"Class {pred_idx}"

                    # Kartu Hasil Diagnosis Menarik
                    st.markdown(f"""
                    <div class="diagnosis-header">
                        <div>
                            <div class="diagnosis-label">HASIL DETEKSI MODEL</div>
                            <div class="diagnosis-value">{result_name}</div>
                        </div>
                        <div class="confidence-badge">
                            <div style="font-size:0.75rem; opacity:0.85;">CONFIDENCE</div>
                            <div class="confidence-val">{confidence:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(min(int(confidence), 100))

                    # Grafik Distribusi Probabilitas
                    with st.expander("📊 Lihat Sebaran Probabilitas Semua Kelas", expanded=False):
                        chart_data = {CLASS_NAMES[i]: float(preds[0][i] * 100) for i in range(len(CLASS_NAMES))}
                        st.bar_chart(chart_data)

                    # Analisis Mendalam dari LLM
                    if not gemini_api_key:
                        st.warning("⚠️ API Key Gemini belum diset di file `.env` untuk menghasilkan insight medis dokter AI.")
                    else:
                        with st.spinner("🤖 Dokter AI (Gemini) sedang menyusun protokol medis dan panduan penanganan..."):
                            chart_data = {CLASS_NAMES[i]: float(preds[0][i] * 100) for i in range(len(CLASS_NAMES))}
                            llm_output = generate_llm_analysis(
                                api_key=gemini_api_key,
                                disease_name=result_name,
                                confidence=confidence,
                                class_scores=chart_data,
                                additional_notes=user_notes
                            )
                            st.markdown(f"""
                            <div class="llm-report">
                                <div class="llm-badge">🩺 LAPORAN MEDIS AI • GEMINI INSIGHT</div>
                                <div>{llm_output}</div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi error saat melakukan prediksi: {e}")
