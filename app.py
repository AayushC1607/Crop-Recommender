import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your pre-trained Random Forest model
model = joblib.load('crop_recommender_model.pkl')

# Define the dictionary of crops
crop_mapping = {
    0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
    5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidney Beans',
    10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Mungbean',
    15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'Pigeonpeas', 19: 'Pomegranate',
    20: 'Rice', 21: 'Watermelon'
}

def validate_numeric_input(value, min_val, max_val, field_name):
    """
    Validate numeric input within a range.
    """
    try:
        value = float(value)
        if value < min_val or value > max_val:
            st.error(f"{field_name} should be between {min_val} and {max_val}.")
            return None
        return value
    except ValueError:
        st.error(f"Invalid input for {field_name}. Please enter a numeric value.")
        return None

# Streamlit app
st.title("Crop Recommendation System")

# Input fields with validation
N = st.text_input('Nitrogen (N)', '')
P = st.text_input('Phosphorus (P)', '')
K = st.text_input('Potassium (K)', '')
temperature = st.text_input('Temperature (°C)', '')
humidity = st.text_input('Humidity (%)', '')
ph = st.text_input('pH of soil', '')
rainfall = st.text_input('Rainfall (mm)', '')

# Validate inputs with their respective ranges
N = validate_numeric_input(N, 0, 140, 'Nitrogen (N)')
P = validate_numeric_input(P, 5, 145, 'Phosphorus (P)')
K = validate_numeric_input(K, 5, 205, 'Potassium (K)')
temperature = validate_numeric_input(temperature, 0, 50, 'Temperature (°C)')
humidity = validate_numeric_input(humidity, 10, 100, 'Humidity (%)')
ph = validate_numeric_input(ph, 3.5, 9.94, 'pH of soil')
rainfall = validate_numeric_input(rainfall, 15, 300, 'Rainfall (mm)')

# Check for invalid inputs
if None in [N, P, K, temperature, humidity, ph, rainfall]:
    st.error("Please correct the invalid inputs before proceeding.")
else:
    if st.button('Recommend'):
        # Prepare input for model
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]
        
        # Map prediction to crop
        recommended_crop = crop_mapping.get(prediction, 'Unknown')
        
        st.write(f"Recommended Crop: {recommended_crop}")

        
        # Map prediction to crop
        recommended_crop = crop_mapping.get(prediction, 'Unknown')
        
        st.write(f"Recommended Crop: {recommended_crop}")
