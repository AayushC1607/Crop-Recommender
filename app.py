import streamlit as st
import pandas as pd
import joblib

model = joblib.load('')

crops = ["Apple", "Banana", "Blackgram", "Chickpea", "Coconut", "Coffee", 
    "Cotton", "Grapes", "Jute", "Kidney Beans", "Lentil", "Maize", 
    "Mango", "Moth Beans", "Moong Beans", "Muskmelon", "Orange", 
    "Papaya", "Pigeonpea", "Pomogrenate", "Rice", "Watermelon"]

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_data)
    return crops[prediction[0]]

st.title("Crop Recommendation System")

N = st.number_input('Nitrogen (N)', min_value=0)
P = st.number_input('Phosphorus (P)', min_value=0)
K = st.number_input('Potassium (K)', min_value=0)
temperature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100)
ph = st.number_input('pH of soil', min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input('Rainfall (mm)', min_value=0)

if st.button('Predict'):
    crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.write(f"Recommended Crop: {crop}")
