import numpy as np
import streamlit as st

from web_functions import predict

def app(df, x, y):

    st.title("Halaman Prediksi")
    Pregnancies = st.text_input('Input Nilai Pregnancies 0-11' 0,11)
    Glucose = st.text_input('Input Nilai Glucose')
    BloodPressure = st.text_input('Input Nilai Blood Pressure (mm Hg)')
    SkinThickness = st.text_input('Input Nilai SkinThickness (mm)')
    Insulin = st.text_input('Input Nilai Insulin (mu U/ml)')
    BMI = st.text_input('Input Nilai BMI (berat badan dalam kg/(tinggi badan dalam m)^2)')
    DiabetesPedigreeFunction = st.text_input('Input Nilai Diabetes Pedigree Function')
    Age = st.text_input('Input Nilai Age (dalam tahun)')

    features = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    # Handle empty inputs
    try:
        # Konversi input text ke numeric dengan penanganan error
        features_numeric = np.array([float(f) for f in features if f])  # Hanya konversi jika nilai tidak kosong
    except ValueError:
        st.error("Mohon isi semua input dengan nilai numerik yang valid.")
        return  # Hentikan proses jika ada nilai kosong

    # Tombol Prediksi
    if st.button("Prediksi"):
        prediction, score = predict(x, y, features_numeric)
        score = score
        st.info("Prediksi Sukses...")

        if prediction == 1:
            st.warning("Pasien terkena diabetes")
        else:
            st.success("Pasien tidak terkena diabetes")

        st.write("Model yang digunakan memiliki tingkat akurasi ", (score * 100), "%")
