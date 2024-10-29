import numpy as np
import pickle
import streamlit as st
from bs4 import BeautifulSoup

# Tải mô hình từ file lr_model.pkl
with open(r"./lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Tải scaler đã huấn luyện
with open(r"./scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


def predict_typhoon(inputs):
    # Scale and predict
    inputs_scaled = scaler.transform(inputs)
    prediction = lr_model.predict(inputs_scaled)
    
    return prediction[0][0]

# Streamlit App
st.title("Typhoon Prediction App")
st.title("THPT Mỹ Hào - Đặng Chiến Hồng Quang and Nguyễn Kim Vũ" )

# Form for input
with st.form("typhoon_form"):
    st.write("### Input Data for Time Step 1")
    # Inputs for Time Step 1
    latitude1 = st.number_input("Latitude", value=11.0)
    longitude1 = st.number_input("Longitude", value=12.0)
    year1 = st.number_input("Year", value=2021)
    month1 = st.number_input("Month", value=12)
    day1 = st.number_input("Day", value=25)
    hour1 = st.number_input("Hour", value=11)
    grade1 = st.number_input("Grade", value=2)
    pressure1 = st.number_input("Pressure", value=3)
    wind_speed1 = st.number_input("Wind Speed", value=13)
    radius_50kt1 = st.number_input("Radius 50kt", value=31)
    radius_30kt1 = st.number_input("Radius 30kt", value=13)
    shortest_radius_50kt1 = st.number_input("Shortest Radius 50kt", value=1)
    shortest_radius_30kt1 = st.number_input("Shortest Radius 30kt", value=3)

    st.write("### Input Data for Time Step 2")
    # Inputs for Time Step 2
    latitude2 = st.number_input("Latitude (Time Step 2)", value=11.0)
    longitude2 = st.number_input("Longitude (Time Step 2)", value=12.0)
    year2 = st.number_input("Year (Time Step 2)", value=2021)
    month2 = st.number_input("Month (Time Step 2)", value=12)
    day2 = st.number_input("Day (Time Step 2)", value=25)
    hour2 = st.number_input("Hour (Time Step 2)", value=11)
    grade2 = st.number_input("Grade (Time Step 2)", value=2)
    pressure2 = st.number_input("Pressure (Time Step 2)", value=3)
    wind_speed2 = st.number_input("Wind Speed (Time Step 2)", value=13)
    radius_50kt2 = st.number_input("Radius 50kt (Time Step 2)", value=31)
    radius_30kt2 = st.number_input("Radius 30kt (Time Step 2)", value=13)
    shortest_radius_50kt2 = st.number_input("Shortest Radius 50kt (Time Step 2)", value=1)
    shortest_radius_30kt2 = st.number_input("Shortest Radius 30kt (Time Step 2)", value=3)
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# Prediction and Output
if submit_button:
    # Collecting inputs into a single array
    inputs = np.array([
        latitude1, longitude1, year1, month1, day1, hour1, grade1, pressure1, wind_speed1,
        radius_50kt1, radius_30kt1, shortest_radius_50kt1, shortest_radius_30kt1,
        latitude2, longitude2, year2, month2, day2, hour2, grade2, pressure2, wind_speed2,
        radius_50kt2, radius_30kt2, shortest_radius_50kt2, shortest_radius_30kt2
    ]).reshape(1, -1)

    # Prediction
    prediction = predict_typhoon(inputs)
    threshold = 30  # Define your threshold for typhoon

    if prediction > threshold:
        st.write("### Prediction: Likely a Typhoon")
    else:
        st.write("### Prediction: Not a Typhoon")

    st.write(f"Predicted Wind Speed: {prediction:.2f}")

#html
with open('index.html', 'r', encoding='utf-8') as file:
    content = file.read()

soup = BeautifulSoup(content, 'html.parser')
print(soup.prettify())