# Import Library
import streamlit as st
import pandas as pd
from web_functions import load_data
from Tabs import home, predict, visualise

Tabs = {
    "Home": home,
    "Prediction": predict,
    "Visualisation": visualise
}

# Membuat Sidebar
st.sidebar.title("Navigasi")

# Membuat radio option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# load dataset
df, x, y = load_data()

# Kondisi call app function
if page in ["Prediction", "Visualisation"]:
    df, x, y = load_data()
    Tabs[page].app(df, x, y)
else:
    Tabs[page].app()
