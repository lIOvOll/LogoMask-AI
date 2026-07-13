import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Sensor Video", page_icon="🎥", layout="wide")

# Load Global CSS
css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Session State Initialization
if 'tahap' not in st.session_state:
    st.session_state.tahap = 'upload' 
if 'video_hasil' not in st.session_state:
    st.session_state.video_hasil = None
if 'waktu_proses' not in st.session_state:
    st.session_state.waktu_proses = 0
if 'total_frame' not in st.session_state:
    st.session_state.total_frame = 0

# Model Initialization
@st.cache_resource
def load_model():
    path_model = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    return YOLO(path_model)

model = load_model()
daftar_merek = list(model.names.values())

st.title("🎥 Sensor Video")
st.markdown("---")

# Parameters Configuration
st.markdown("#### ⚙️ Konfigurasi Parameter")
col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    st.session_state.conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
with col_slider2:
    tingkat_blur = st.slider("Intensitas Blur", 1, 10, 5, 1)
    st.session_state.blur_kernel = (tingkat_blur * 20) - 1

st.markdown("#### 🎯 Filter Kelas")
st.session_state.merek_terpilih = st.multiselect(
    "Pilih kelas merek untuk operasi sensor:",
    options=daftar_merek,
    default=daftar_merek
)
st.markdown("---")

# File Upload Handler
if st.session_state.tahap == 'upload':
    st.markdown("#### 📤 Unggah Video")
    uploaded_file = st.file_uploader("", type=['mp4', 'avi'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Proses Video", type="primary", use_container_width=True):
            if not st.session_state.merek_terpilih:
                st.warning("⚠️ Diperlukan minimal satu kelas target.")
            else:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.tahap = 'proses' 
                st.rerun() 

# Processing Handler
elif st.session_state.tahap == 'proses':
    uploaded_file = st.session_state.uploaded_file
    
    with st.status("⏳ Memproses pipeline...", expanded=True) as status:
        my_bar = st.progress(0)
        teks_status = st.empty()
        start_time = time.time()
        
        # Temp File Setup
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.getvalue())
        temp_input.close()
        
        temp_output_raw = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_raw.close()

        # OpenCV Video Capture
        cap = cv2.VideoCapture(temp_input.name)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps, total_frames = int(cap.get(5)), int(cap.get(7))

        out = cv2.VideoWriter(temp_output_raw.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0
        conf_thresh = st.session_state.conf_threshold
        b_kernel = st.session_state.blur_kernel
        merek_target = st.session_state.merek_terpilih 

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            persist_val = False if frame_count == 0 else True
            results = model.track(frame, conf=conf_thresh, persist=persist_val, tracker="bytetrack.yaml", verbose=False)
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    nama_terdeteksi = model.names[cls_id]
                    
                    if nama_terdeteksi in merek_target:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.shape[0] > 0 and roi.shape[1] > 0:
                            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (b_kernel, b_kernel), 30)

            out.write(frame)
            frame_count += 1
            my_bar.progress(min(frame_count / total_frames, 1.0))
            teks_status.text(f"Memproses frame: {frame_count} / {total_frames}")

        cap.release()
        out.release()
        
        st.session_state.waktu_proses = round(time.time() - start_time, 2)
        st.session_state.total_frame = frame_count
        status.update(label="Selesai", state="complete", expanded=False)

    with open(temp_output_raw.name, 'rb') as f:
        st.session_state.video_hasil = f.read()
        
    os.unlink(temp_input.name)
    os.unlink(temp_output_raw.name)
    
    st.session_state.tahap = 'hasil'
    st.rerun()

# Output View
elif st.session_state.tahap == 'hasil':
    st.markdown("#### 🎉 Output")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Waktu Eksekusi", f"{st.session_state.waktu_proses} s")
    col_m2.metric("Total Frame", st.session_state.total_frame)
    col_m3.metric("Kecepatan Proses", f"{round(st.session_state.total_frame / st.session_state.waktu_proses, 1) if st.session_state.waktu_proses > 0 else 0} FPS")
    
    st.video(st.session_state.video_hasil)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download", st.session_state.video_hasil, "output_video.mp4", "video/mp4", use_container_width=True, type="primary")
    with col2:
        if st.button("Proses Baru", use_container_width=True):
            st.session_state.tahap = 'upload'
            st.rerun()