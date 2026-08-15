import streamlit as st
import numpy as np
import joblib
import pandas as pd
import random

# ── Load Files Safely ─────────────────────────────────────
def load_file(path):
    return joblib.load(path)

model = load_file(r"D:\expert project\best_co2_emission_model.pkl")
fuel_encoder = load_file(r"D:\expert project\fuel_encoder.pkl")
transmission_encoder = load_file(r"D:\expert project\transmission_encoder.pkl")
vehicle_class_encoder = load_file(r"D:\expert project\vehicle_class_encoder.pkl")
scaler = load_file(r"D:\expert project\scalerf.pkl")

# ── Mapping ─────────────────────────────────────────────
co2_map = {0: "High", 1: "Low", 2: "Medium"}

# ── Page Config ─────────────────────────────────────────
st.set_page_config(page_title="CO2 Emission AI", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #111;
        box-shadow: 0 0 10px rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚗 CO2 Emission AI Dashboard</p>', unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("### 🔧 Vehicle Inputs")

    # Random generator
    if st.button("🎲 Generate Random Test"):
        vehicle_class = random.choice(list(vehicle_class_encoder.classes_))
        engine_size = round(random.uniform(1.0, 6.0), 1)
        cylinders = random.choice([3,4,6,8])
        transmission = random.choice(list(transmission_encoder.classes_))
        fuel_type = random.choice(list(fuel_encoder.classes_))
        fuel_city = round(random.uniform(6, 18), 1)
        fuel_hwy = round(random.uniform(5, 14), 1)
        fuel_comb = round((fuel_city + fuel_hwy)/2, 1)
    else:
        vehicle_class = st.selectbox("Vehicle Class", vehicle_class_encoder.classes_)
        engine_size = st.slider("Engine Size", 1.0, 8.0, 2.0)
        cylinders = st.selectbox("Cylinders", [3,4,6,8,12])
        transmission = st.selectbox("Transmission", transmission_encoder.classes_)
        fuel_type = st.selectbox("Fuel Type", fuel_encoder.classes_)
        fuel_city = st.slider("City Consumption", 5.0, 25.0, 10.0)
        fuel_hwy = st.slider("Highway Consumption", 4.0, 20.0, 8.0)
        fuel_comb = (fuel_city + fuel_hwy) / 2

with col2:
    st.markdown("### 📊 Live Data Visualization")

    df = pd.DataFrame({
        "Metric": ["City", "Highway", "Combined"],
        "Consumption": [fuel_city, fuel_hwy, fuel_comb]
    })

    st.bar_chart(df.set_index("Metric"))

# ── Prediction ─────────────────────────────────────────
if st.button("🚀 Predict Emission"):

    vehicle_class_enc = vehicle_class_encoder.transform([vehicle_class])[0]
    transmission_enc = transmission_encoder.transform([transmission])[0]
    fuel_type_enc = fuel_encoder.transform([fuel_type])[0]

    features = np.array([[
        vehicle_class_enc,
        engine_size,
        cylinders,
        transmission_enc,
        fuel_type_enc,
        fuel_city,
        fuel_hwy,
        fuel_comb
    ]])

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    result = co2_map[pred]

    st.markdown("---")
    st.markdown("## 🧠 Prediction Result")

    if result == "High":
        st.error("🚨 High CO2 Emission")
        explanation = "This vehicle consumes a lot of fuel and produces high emissions."
        tips = [
            "Switch to hybrid or electric",
            "Reduce engine size",
            "Use eco-driving habits"
        ]

    elif result == "Medium":
        st.warning("⚖️ Medium CO2 Emission")
        explanation = "Moderate fuel efficiency with average emissions."
        tips = [
            "Maintain tire pressure",
            "Drive smoothly",
            "Reduce unnecessary load"
        ]

    else:
        st.success("🌱 Low CO2 Emission")
        explanation = "Fuel-efficient vehicle with low environmental impact."
        tips = [
            "Keep regular maintenance",
            "Continue eco-driving",
            "Consider electric upgrade"
        ]

    st.write("### 🌍 Explanation")
    st.info(explanation)

    st.write("### 💡 Recommendations")
    for tip in tips:
        st.write(f"✔️ {tip}")