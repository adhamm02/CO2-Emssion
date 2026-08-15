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

# Initialize session state for inputs so that they persist and can be updated by the random button
if "vehicle_class" not in st.session_state:
    st.session_state.vehicle_class = vehicle_class_encoder.classes_[0]
    st.session_state.engine_size = 2.0
    st.session_state.cylinders = 4
    st.session_state.transmission = transmission_encoder.classes_[0]
    st.session_state.fuel_type = fuel_encoder.classes_[0]
    st.session_state.fuel_city = 10.0
    st.session_state.fuel_hwy = 8.0

def generate_random():
    st.session_state.vehicle_class = random.choice(list(vehicle_class_encoder.classes_))
    st.session_state.engine_size = float(round(random.uniform(1.0, 6.0), 1))
    st.session_state.cylinders = int(random.choice([3, 4, 6, 8, 12]))
    st.session_state.transmission = random.choice(list(transmission_encoder.classes_))
    st.session_state.fuel_type = random.choice(list(fuel_encoder.classes_))
    st.session_state.fuel_city = float(round(random.uniform(6.0, 18.0), 1))
    st.session_state.fuel_hwy = float(round(random.uniform(5.0, 14.0), 1))

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 🔧 Vehicle Configuration")
    
    # Random generator updates the session state via the callback
    st.button("🎲 Generate Random Test", on_click=generate_random, use_container_width=True)
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            vehicle_class = st.selectbox("Vehicle Class", vehicle_class_encoder.classes_, key="vehicle_class")
            engine_size = st.slider("Engine Size (L)", 1.0, 8.0, key="engine_size", step=0.1)
            cylinders = st.selectbox("Cylinders", [3, 4, 6, 8, 12], key="cylinders")
        with c2:
            transmission = st.selectbox("Transmission", transmission_encoder.classes_, key="transmission")
            fuel_type = st.selectbox("Fuel Type", fuel_encoder.classes_, key="fuel_type")
            fuel_city = st.slider("City Consumption (L/100km)", 5.0, 25.0, key="fuel_city", step=0.1)
            fuel_hwy = st.slider("Highway Consumption (L/100km)", 4.0, 20.0, key="fuel_hwy", step=0.1)
            
        fuel_comb = (fuel_city + fuel_hwy) / 2
        st.metric("Estimated Combined Consumption", f"{fuel_comb:.1f} L/100km")

with col2:
    st.markdown("### 📊 Fuel Consumption Data")
    
    df = pd.DataFrame({
        "Metric": ["City", "Highway", "Combined"],
        "Consumption (L/100km)": [fuel_city, fuel_hwy, fuel_comb]
    })
    st.bar_chart(df.set_index("Metric"), color="#ff4b4b")

    st.markdown("---")
    st.markdown("### 🚀 AI Prediction")
    predict_btn = st.button("Predict Emission Level", type="primary", use_container_width=True)

# ── Prediction ─────────────────────────────────────────
if predict_btn:
    vehicle_class_enc = vehicle_class_encoder.transform([vehicle_class])[0]
    transmission_enc = transmission_encoder.transform([transmission])[0]
    fuel_type_enc = fuel_encoder.transform([fuel_type])[0]

    features = pd.DataFrame([[
        vehicle_class_enc,
        engine_size,
        cylinders,
        transmission_enc,
        fuel_type_enc,
        fuel_city,
        fuel_hwy,
        fuel_comb
    ]], columns=[
        'Vehicle Class', 'Engine Size(L)', 'Cylinders', 'Transmission', 'Fuel Type',
        'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)',
        'Fuel Consumption Comb (L/100 km)'
    ])

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    result = co2_map[pred]

    st.markdown("---")
    st.markdown("## 🧠 Analysis Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if result == "High":
            st.error("🚨 **HIGH** Emission")
        elif result == "Medium":
            st.warning("⚖️ **MEDIUM** Emission")
        else:
            st.success("🌱 **LOW** Emission")

    with res_col2:
        if result == "High":
            st.write("**Explanation:** This vehicle consumes a lot of fuel and produces high emissions.")
            st.info("💡 **Recommendations:**\n- Switch to hybrid or electric\n- Reduce engine size\n- Use eco-driving habits")
        elif result == "Medium":
            st.write("**Explanation:** Moderate fuel efficiency with average emissions.")
            st.info("💡 **Recommendations:**\n- Maintain tire pressure\n- Drive smoothly\n- Reduce unnecessary load")
        else:
            st.write("**Explanation:** Fuel-efficient vehicle with low environmental impact.")
            st.info("💡 **Recommendations:**\n- Keep regular maintenance\n- Continue eco-driving\n- Consider electric upgrade")
