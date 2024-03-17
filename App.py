import streamlit as st
import pandas as pd
import pickle
import numpy as np


# Load the pickled model
with open('home_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the location encoder
with open('location_encoder.pkl', 'rb') as file:
    location_encoder = pickle.load(file)

location_df = pd.read_csv('location.csv')
location_columns = location_df['location'].tolist()

st.title(':moneybag: Bengaluru House Price Prediction :house_buildings:')

st.write('Enter details of your property to predict its estimated price: :man-tipping-hand:')

st.markdown("""
        <style>
               .block-container {
                    padding-left: 1rem;
                    padding-right:1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Create columns for user input
col1, col2, col3 = st.columns((0.33, 0.33, 0.33))

with col1:
    bhk = st.number_input('BHK', min_value=1, max_value=10)

with col2:
    bath = st.number_input('Bath', min_value=1, max_value=10)

with col3:
    balcony = st.number_input('Balcony', min_value=1, max_value=10)

col4, col5 = st.columns((0.5, 0.5))

with col4:
    location = st.selectbox('Location', location_columns)

with col5:
    total_sqrt = st.number_input('Area in Square Foot', min_value=100)

# Encode the location using the loaded encoder
encoded_location = location_encoder.transform([location])[0]

ok = st.button('Estimate Price :dollar:')

if ok:
    # Prepare input data as a numpy array
    input_data = pd.DataFrame({
        'total_sqft': [total_sqrt],
        'bath': [bath],
        'balcony': [balcony],
        'bhk': [bhk],
        'location_encoded': [encoded_location]
    })

    predicted_price = model.predict(input_data)[0]

    if predicted_price > 0:
        st.success(f'The estimated price for your property is approximately ₹ {predicted_price:0.02f} lakhs.')
    else:
        st.error('There may be an error with the prediction. Please check your input values or contact the developer.')
