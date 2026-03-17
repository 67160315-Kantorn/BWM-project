import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="BMW Price Predictor", page_icon="🚗")

st.title("🚗 BMW Used Car Price Predictor")
st.write("ทำนายราคามือสองของ BMW ด้วย Machine Learning")

# =========================
# MODEL CONFIG
# =========================
MODEL_PATH = "model_artifacts/bmw_price_pipeline.pkl"

# 🔥 ใส่ลิงก์โมเดลของมึงตรงนี้
MODEL_URL = "https://drive.google.com/uc?export=download&id=1JsP2pnQ8smLhspYD4M-y1Dx9mXeppxqU"

# =========================
# DOWNLOAD MODEL IF NOT EXISTS
# =========================
@st.cache_resource
def load_model():
    os.makedirs("model_artifacts", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.warning("📥 กำลังดาวน์โหลดโมเดลครั้งแรก... (รอแปปนึง)")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# =========================
# INPUT UI
# =========================
st.header("🔧 ใส่ข้อมูลรถ")

year = st.number_input("ปีรถ", min_value=1990, max_value=2025, value=2018)
mileage = st.number_input("ระยะทาง (km)", min_value=0, max_value=500000, value=50000)
engine_size = st.number_input("ขนาดเครื่องยนต์ (L)", min_value=1.0, max_value=5.0, value=2.0)
fuel_type = st.selectbox("ประเภทเชื้อเพลิง", ["Petrol", "Diesel", "Hybrid", "Electric"])
transmission = st.selectbox("เกียร์", ["Automatic", "Manual"])

# =========================
# PREDICT
# =========================
if st.button("💰 ทำนายราคา"):
    try:
        input_df = pd.DataFrame({
            "year": [year],
            "mileage": [mileage],
            "engineSize": [engine_size],
            "fuelType": [fuel_type],
            "transmission": [transmission]
        })

        prediction = model.predict(input_df)[0]

        st.success(f"💵 ราคาประมาณ: {prediction:,.0f} บาท")

    except Exception as e:
        st.error("❌ เกิด error ในการทำนาย")
        st.exception(e)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
