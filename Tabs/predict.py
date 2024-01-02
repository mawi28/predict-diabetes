import numpy as np
import streamlit as st

from web_functions import predict

def app(df, x, y):

    st.title("Halaman Prediksi")
    Pregnancies = st.number_input('Input Nilai Pregnancies 0-17', 0,17)
    Glucose = st.number_input('Input Nilai Glucose 0-199', 0,199)
    BloodPressure = st.number_input('Input Nilai Blood Pressure (mm Hg 0-122)', 0,122)
    SkinThickness = st.number_input('Input Nilai SkinThickness (mm 0-99)', 0,99)
    Insulin = st.number_input('Input Nilai Insulin (mu U/ml 0-846)', 0,846)
    BMI = st.number_input('Input Nilai BMI 0 - 67.1 (berat badan dalam kg/(tinggi badan dalam m)^2)', 0.00,67.1)
    DiabetesPedigreeFunction = st.number_input('Input Nilai Diabetes Pedigree Function 0.078-2.420', 0.078, 2.420)
    Age = st.text_input('Input Nilai Age (dalam tahun) 21-81', 21, 81)

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
