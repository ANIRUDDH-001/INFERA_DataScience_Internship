import streamlit as st
import pandas as pd
import joblib

# ===============================
# 🔹 Setup
# ===============================
st.set_page_config(page_title="🌤️ Weather Temperature Predictor", layout="centered")
st.title("🌤️ Weather Max Temperature Predictor")
st.markdown("Enter current weather parameters to predict the **Maximum Temperature (°C)**.")

# ===============================
# 🔹 Load Trained Model
# ===============================
model = joblib.load("models/best_model.pkl")

# ===============================
# 🔹 User Input Fields
# ===============================
with st.form("weather_form"):
    st.subheader("📥 Weather Inputs")

    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0, step=0.1)
        evaporation = st.number_input("Evaporation (mm)", 0.0, 50.0, 0.0, step=0.1)
        sunshine = st.number_input("Sunshine (hours)", 0.0, 24.0, 0.0, step=0.1)
        gust_speed = st.number_input("Max Wind Gust Speed (km/h)", 0.0, 150.0, 20.0, step=1.0)
    with col2:
        temp_9am = st.number_input("9AM Temperature (°C)", -5.0, 45.0, 20.0, step=0.1)
        humidity_9am = st.slider("9AM Humidity (%)", 0, 100, 70)
        pressure_9am = st.number_input("9AM Pressure (hPa)", 980.0, 1050.0, 1015.0, step=0.1)
        wind_3pm = st.number_input("3PM Wind Speed (km/h)", 0.0, 100.0, 10.0, step=1.0)

    humidity_3pm = st.slider("3PM Humidity (%)", 0, 100, 55)
    pressure_3pm = st.number_input("3PM Pressure (hPa)", 980.0, 1050.0, 1012.0, step=0.1)

    submitted = st.form_submit_button("🔍 Predict Maximum Temperature")

# ===============================
# 🔹 Prediction
# ===============================
# Load the trained model and expected feature columns
model = joblib.load('models/best_model.pkl')
feature_columns = joblib.load('models/model_features.pkl')  # <-- NEW

# Prepare user input
input_data = pd.DataFrame([{
    'Rainfall (mm)': rainfall,
    'Evaporation (mm)': evaporation,
    'Sunshine (hours)': sunshine,
    'Speed of maximum wind gust (km/h)': gust_speed,
    '9am Temperature (°C)': temp_9am,
    '9am relative humidity (%)': humidity_9am,
    '9am MSL pressure (hPa)': pressure_9am,
    '3pm Temperature (°C)': 25.0,  # Default/placeholder values
    '3pm relative humidity (%)': humidity_3pm,
    '3pm MSL pressure (hPa)': pressure_3pm,
    '3pm wind speed (km/h)': wind_3pm,
    '3pm cloud amount (oktas)': 5,
    '3pm wind direction_NE': 1,  # Example for one-hot encoded direction
    # Add more if needed
}])

# Add all missing columns with 0
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Predict
if submitted:
    prediction = model.predict(input_data)[0]
    st.success(f"🌡️ Predicted Max Temperature: **{prediction:.2f} °C**")


# ===============================
# 🔹 Footer
# ===============================
st.markdown("---")
st.caption("📊 Powered by Random Forest Regression | Built with ❤️ using Streamlit")
