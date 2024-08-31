import streamlit as st
import pickle
import cv2

with open('D:\SIH', 'rb') as f:
    model = pickle.load(f)

uploaded_image = st.file_uploader("Upload an Image")

if uploaded_image is not None:
    img = cv2.imread(uploaded_image)

    img = cv2.resize(img, (224, 224))

    img = img.reshape(1, 224, 224, 3)

    prediction = model.predict(img)

    st.write("Predicted Crop Disease:", predicted_disease)