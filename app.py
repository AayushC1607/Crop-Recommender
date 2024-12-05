from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained Random Forest model
model = joblib.load('crop_recommender_model.pkl')

# Define the dictionary of crops
crop_mapping = {
    0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
    5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidney Beans',
    10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Mungbean',
    15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'Pigeonpeas', 19: 'Pomegranate',
    20: 'Rice', 21: 'Watermelon'
}

app = Flask(__name__)

@app.route('/recommend-crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    
    # Extract values from the request
    try:
        N = float(data['nitrogen'])
        P = float(data['phosphorus'])
        K = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
    except ValueError:
        return jsonify({"error": "Invalid input, all fields must be numeric"}), 400

    # Prepare input for the model
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Map prediction to crop
    recommended_crop = crop_mapping.get(prediction, 'Unknown')
    
    # Return the recommended crop as JSON
    return jsonify({"recommended_crop": recommended_crop})

if __name__ == '__main__':
    app.run(debug=True)
