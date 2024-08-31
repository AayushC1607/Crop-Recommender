import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
with open('', 'rb') as model_file:
    model = pickle.load(model_file)

# Define crop mapping
crop_mapping = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate',
    20: 'rice', 21: 'watermelon'
}

# Streamlit app
def main():
    st.title('Crop Recommendation System')

    # Input fields
    N = st.number_input('Nitrogen (N)', min_value=0.0, format="%.2f")
    P = st.number_input('Phosphorus (P)', min_value=0.0, format="%.2f")
    K = st.number_input('Potassium (K)', min_value=0.0, format="%.2f")
    temperature = st.number_input('Temperature (°C)', min_value=-50.0, max_value=50.0, format="%.2f")
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, format="%.2f")
    ph = st.number_input('pH Level', min_value=0.0, max_value=14.0, format="%.2f")
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, format="%.2f")

    # Debugging output
    st.write(f"N: {N}, P: {P}, K: {K}, Temperature: {temperature}, Humidity: {humidity}, pH: {ph}, Rainfall: {rainfall}")

    # Make prediction when button is pressed
    if st.button('Recommend'):
        try:
            # Prepare input for model
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)[0]
            
            # Map prediction to crop
            recommended_crop = crop_mapping.get(prediction, 'Unknown')
            
            st.write(f"Recommended Crop: {recommended_crop}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
