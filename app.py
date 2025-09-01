import streamlit as st
import pandas as pd
import pickle

# Load model
with open("x_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌞 Solar Power Generation Prediction App")
st.write("Enter the input values to predict power generation.")

# Example input fields (replace with your dataset’s features)
distance_to_solar_noon = st.number_input("Distance to Solar Noon", value=1.0)
temperature = st.number_input("Temperature (°C)", value=25.0)
wind_direction = st.number_input("Wind Direction (degrees)", value=180.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)
sky_cover = st.number_input("Sky Cover (%)", value=50.0)
visibility = st.number_input("Visibility (km)", value=10.0)
humidity = st.number_input("Humidity (%)", value=60.0)
avg_wind_speed = st.number_input("Average Wind Speed (period)", value=3.0)
avg_pressure = st.number_input("Average Pressure (period)", value=1013.0)

if st.button("Predict"):
    # Arrange inputs as a DataFrame for the model
    input_data = pd.DataFrame([[distance_to_solar_noon, temperature, wind_direction,
                                 wind_speed, sky_cover, visibility, humidity,
                                 avg_wind_speed, avg_pressure]],
                               columns=['distance-to-solar-noon', 'temperature', 'wind-direction',
                                        'wind-speed', 'sky-cover', 'visibility', 'humidity',
                                        'average-wind-speed-(period)', 'average-pressure-(period)'])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Power Generated: {prediction:.2f} kW")
