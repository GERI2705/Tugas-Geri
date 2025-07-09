import streamlit as st
from skimage import io, color, filters, feature, exposure, img_as_ubyte
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deteksi Tepi Citra", layout="wide")
st.title("🖼️ Deteksi Tepi Citra Digital")
st.write("Pilih metode deteksi tepi dan lihat hasilnya secara visual.")

uploaded_file = st.file_uploader("Unggah Gambar", type=["png", "jpg", "jpeg"])
metode = st.selectbox("Pilih Metode Deteksi Tepi", ["Sobel", "Canny"])
tampilan = st.radio("Tampilkan sebagai", ["Biner (0/255)", "Grayscale (normalisasi)"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    gray = color.rgb2gray(img_array)

    # ✨ Perkuat kontras dengan histogram equalization
    gray_enhanced = exposure.equalize_adapthist(gray, clip_limit=0.03)

    if metode == "Canny":
        edges = feature.canny(gray_enhanced, sigma=1.0)
        edges_display = edges.astype(np.uint8) * 255  # Biner

    elif metode == "Sobel":
        sobel_result = filters.sobel(gray_enhanced)
        if tampilan == "Biner (0/255)":
            edges = sobel_result > 0.01  # Bisa disesuaikan
            edges_display = edges.astype(np.uint8) * 255
        else:
            rescaled = exposure.rescale_intensity(sobel_result, in_range='image', out_range=(0, 1))
            edges_display = img_as_ubyte(rescaled)

    # Hitung piksel putih
    count_white = int(np.sum(edges_display == 255))

    # Tampilkan
    st.image(img, caption="🖼️ Gambar Asli", use_container_width=True)
    st.image(edges_display, caption=f"📌 Hasil Deteksi Tepi ({metode})", use_container_width=True, clamp=True)
    st.success(f"Jumlah Piksel Putih (Tepi): {count_white}")
