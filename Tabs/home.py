import streamlit as st
import matplotlib.image as mpimg

def app():
    st.title("Aplikasi Prediksi Penyakit Diabetes")

    # Memuat gambar
    image_path = 'diabet.jpg'
    image = mpimg.imread(image_path)

    st.text("Selamat datang di aplikasi prediksi penyakit diabetes!")
    st.image(image, width=500)  

# Jalankan aplikasi
if __name__ == '__main__':
    app()
