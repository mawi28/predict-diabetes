# Import Modul
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.cache_data ()
def load_data():

    #Load Dataset
    df = pd.read_csv('diabetes.csv')

    x = df[['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin', 'BMI',	'DiabetesPedigreeFunction',	'Age']]
    y = df[['Outcome']]

    return df, x, y

st.cache_data ()
def train_model(x,y):
    model=KNeighborsClassifier(n_neighbors=14)
        
    model.fit(x,y)

    score = model.score(x,y)

    return model, score

def predict(x,y, features):
    model, score = train_model(x,y)
    
    prediction = model.predict(np.array(features).reshape(1,-1))

    return prediction, score
    