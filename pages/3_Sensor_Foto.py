import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Sensor Foto", page_icon="🖼️", layout="wide")

# Load Global CSS
css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Model Initialization
@st.cache_resource
def load_model():
    path_model = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    return YOLO(path_model)

model = load_model()
daftar_merek = list(model.names.values())

st.title("🖼️ Sensor Foto")
st.markdown("---")

# Parameters Configuration
st.markdown("#### ⚙️ Konfigurasi Parameter")
col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05, key="conf_foto")
with col_slider2:
    tingkat_blur = st.slider("Intensitas Blur", 1, 10, 5, 1, key="blur_foto")
    blur_kernel = (tingkat_blur * 20) - 1

st.markdown("#### 🎯 Filter Kelas")
merek_terpilih = st.multiselect(
    "Pilih kelas merek untuk operasi sensor:",
    options=daftar_merek,
    default=daftar_merek,
    key="merek_foto"
)
st.markdown("---")

# File Upload & Processing Handler
st.markdown("#### 📤 Unggah Foto")
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'webp'], label_visibility="collapsed")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Data Input", use_container_width=True)
    
    if st.button("Proses Foto", type="primary", use_container_width=True):
        if not merek_terpilih:
            st.warning("⚠️ Diperlukan minimal satu kelas target.")
        else:
            with st.spinner("Mengeksekusi prediksi..."):
                results = model.predict(image, conf=conf_threshold, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        nama_terdeteksi = model.names[cls_id]
                        
                        if nama_terdeteksi in merek_terpilih:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                            
                            roi = image[y1:y2, x1:x2]
                            
                            if roi.shape[0] > 0 and roi.shape[1] > 0:
                                image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 30)

                st.success("Selesai")
                
                st.markdown("#### 🎉 Output")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, use_container_width=True)
                
                is_success, buffer = cv2.imencode(".jpg", image)
                io_buf = buffer.tobytes()
                
                st.download_button(
                    label="Download Foto",
                    data=io_buf,
                    file_name="output_foto.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                    type="primary"
                )