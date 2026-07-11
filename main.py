import streamlit as st
import os

# Page Configuration
st.set_page_config(page_title="LogoMask AI", page_icon="🏠", layout="wide")

# Load Global CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page-Specific CSS: Hide Sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 style='text-align: center; color: #2C3E50; margin-top: 20px; margin-bottom: 10px;'>🏠 LogoMask AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555555; margin-bottom: 50px;'>Sistem deteksi dan penyensoran merek berbasis Deep Learning.</p>", unsafe_allow_html=True)

# Navigation Cards
col_kiri, col_card1, col_jarak, col_card2, col_kanan = st.columns([1, 4, 0.5, 4, 1])

with col_card1:
    st.markdown("""
    <div class="card-container">
        <div class="card-icon">🎥</div>
        <div class="card-title">Sensor Video</div>
        <div class="card-text">Deteksi dan penyensoran merek pada file video (MP4/AVI).</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Mode Video ➔", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Sensor_Video.py") 

with col_card2:
    st.markdown("""
    <div class="card-container">
        <div class="card-icon">🖼️</div>
        <div class="card-title">Sensor Foto</div>
        <div class="card-text">Deteksi dan penyensoran merek pada gambar statis (JPG/PNG).</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Mode Foto ➔", use_container_width=True, type="primary"):
        st.switch_page("pages/3_Sensor_Foto.py")