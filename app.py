import streamlit as st
import pandas as pd
import joblib
import json

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="BMW Used Car Price Predictor", page_icon="🚗", layout="centered")

# =========================
# LOAD MODEL + METADATA
# =========================
MODEL_PATH = "model_artifacts/bmw_price_pipeline.pkl"
METADATA_PATH = "model_artifacts/model_metadata.json"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_order = metadata["feature_order"]
choices = metadata["choices"]

model_choices = choices["model"]
transmission_choices = choices["transmission"]
fuel_choices = choices["fuelType"]

# =========================
# TITLE
# =========================
st.title("🚗 BMW Used Car Price Predictor")
st.write("แอปสำหรับทำนายราคารถ BMW มือสองด้วย Machine Learning")

st.markdown("### กรอกข้อมูลรถ")

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Model", model_choices)
    transmission = st.selectbox("Transmission", transmission_choices)
    fuel_type = st.selectbox("Fuel Type", fuel_choices)
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2018, step=1)

with col2:
    mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=50000, step=1000)
    tax = st.number_input("Tax", min_value=0, max_value=1000, value=150, step=10)
    mpg = st.number_input("MPG", min_value=0.0, max_value=300.0, value=50.0, step=0.1)
    engine_size = st.number_input("Engine Size", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

# =========================
# VALIDATION
# =========================
errors = []

if year < 1990 or year > 2025:
    errors.append("Year ไม่สมเหตุสมผล")
if mileage < 0:
    errors.append("Mileage ต้องไม่ติดลบ")
if tax < 0:
    errors.append("Tax ต้องไม่ติดลบ")
if mpg <= 0:
    errors.append("MPG ต้องมากกว่า 0")
if engine_size <= 0:
    errors.append("Engine Size ต้องมากกว่า 0")

for err in errors:
    st.error(err)

# =========================
# PREDICTION
# =========================
if st.button("Predict Price"):
    if errors:
        st.warning("กรุณาแก้ข้อมูลให้ถูกต้องก่อน")
    else:
        input_data = {
            "model": model_name,
            "transmission": transmission,
            "fuelType": fuel_type,
            "year": year,
            "mileage": mileage,
            "tax": tax,
            "mpg": mpg,
            "engineSize": engine_size
        }

        input_df = pd.DataFrame([input_data])

        # เรียงคอลัมน์ให้ตรงกับตอน train
        input_df = input_df[feature_order]

        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)

        # ช่วงประมาณการแบบง่าย
        lower = max(0, prediction * 0.90)
        upper = max(0, prediction * 1.10)

        st.success(f"💰 ราคาที่คาดการณ์: £{prediction:,.2f}")
        st.info(f"ช่วงราคาประมาณ: £{lower:,.2f} - £{upper:,.2f}")

# =========================
# MODEL INFO
# =========================
st.markdown("---")
st.markdown("### 📊 Model Information")
st.write(f"**CV MAE Mean:** {metadata['cv_mae_mean']:.2f}")
st.write(f"**CV MAE Std:** {metadata['cv_mae_std']:.2f}")
st.write(f"**CV R² Mean:** {metadata['cv_r2_mean']:.4f}")

st.markdown("### 📘 Feature Description")
st.write("""
- **Model**: รุ่นของรถ BMW
- **Transmission**: ประเภทเกียร์
- **Fuel Type**: ประเภทเชื้อเพลิง
- **Year**: ปีของรถ
- **Mileage**: ระยะทางที่ใช้งานมาแล้ว
- **Tax**: ภาษีรถ
- **MPG**: อัตราการประหยัดน้ำมัน
- **Engine Size**: ขนาดเครื่องยนต์
""")

st.markdown("### ⚠️ Disclaimer")
st.write("""
โมเดลนี้ใช้เพื่อการประมาณราคาเบื้องต้นเท่านั้น  
ราคาจริงอาจแตกต่างตามสภาพรถ รุ่นย่อย ออปชัน และสภาพตลาดในช่วงเวลานั้น
""")