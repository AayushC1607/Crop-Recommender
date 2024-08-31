import streamlit as st
import pandas as pd
import joblib

# Load your pre-trained model
model = joblib.load('model.pkl')

# Define the list of crops
crops = [
    "Apple", "Banana", "Blackgram", "Chickpea", "Coconut", "Coffee", 
    "Cotton", "Grapes", "Jute", "Kidney Beans", "Lentil", "Maize", 
    "Mango", "Moth Beans", "Moong Beans", "Muskmelon", "Orange", 
    "Papaya", "Pigeonpea", "Pomogrenate", "Rice", "Watermelon"
]

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_data)
    return crops[prediction[0]]

def validate_numeric_input(value):
    try:
        return float(value)
    except ValueError:
        return None

# Streamlit app
st.title("Crop Recommendation System")

# Input fields
N = st.text_input('Nitrogen (N)', '')
P = st.text_input('Phosphorus (P)', '')
K = st.text_input('Potassium (K)', '')
temperature = st.text_input('Temperature (°C)', '')
humidity = st.text_input('Humidity (%)', '')
ph = st.text_input('pH of soil', '')
rainfall = st.text_input('Rainfall (mm)', '')

# Validate inputs
N = validate_numeric_input(N)
P = validate_numeric_input(P)
K = validate_numeric_input(K)
temperature = validate_numeric_input(temperature)
humidity = validate_numeric_input(humidity)
ph = validate_numeric_input(ph)
rainfall = validate_numeric_input(rainfall)

# Check for invalid inputs
if None in [N, P, K, temperature, humidity, ph, rainfall]:
    st.error("Invalid input: Please enter valid numerical values.")
else:
    if st.button('Predict'):
        crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.write(f"Recommended Crop: {crop}")