import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model and scalers
randclf = pickle.load(open('model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# Function to make predictions
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_df = pd.DataFrame(features, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    features_scaled = mx.transform(features_df)
    features_scaled = sc.transform(features_scaled)
    prediction = randclf.predict(features_scaled)
    return prediction[0]

# Streamlit app
st.title("Crop Recommendation System")

N = st.number_input("Enter Nitrogen content (N)", min_value=0, max_value=100, value=50)
P = st.number_input("Enter Phosphorus content (P)", min_value=0, max_value=100, value=50)
K = st.number_input("Enter Potassium content (K)", min_value=0, max_value=100, value=50)
temperature = st.number_input("Enter Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("Enter pH value", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Recommend Crop"):
    result = recommendation(N, P, K, temperature, humidity, ph, rainfall)
    crop_dict = {
        1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya', 7: 'orange',
        8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes', 12: 'mango', 13: 'banana',
        14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 17: 'mungbean', 18: 'mothbeans',
        19: 'pigeonpeas', 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
    }
    recommended_crop = crop_dict[result]
    st.success(f"The recommended crop is: {recommended_crop}")

