import os
import joblib
import gdown
import pandas as pd
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="BMW Price Predictor", page_icon="🚗")

st.title("🚗 BMW Used Car Price Predictor")
st.write("ทำนายราคามือสองของ BMW ด้วย Machine Learning")

# =========================
# MODEL CONFIG
# =========================
MODEL_PATH = "model_artifacts/bmw_price_pipeline.pkl"
FILE_ID = "1JsP2pnQ8smLhspYD4M-y1Dx9mXeppxqU"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    os.makedirs("model_artifacts", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.warning("📥 กำลังดาวน์โหลดโมเดลครั้งแรก... (รอแปปนึง)")
        gdown.download(
            id=FILE_ID,
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

    if not os.path.exists(MODEL_PATH):
        st.error("ไม่พบไฟล์โมเดลหลังดาวน์โหลด")
        st.stop()

    file_size = os.path.getsize(MODEL_PATH)
    if file_size < 1_000_000:
        st.error("ไฟล์โมเดลที่ดาวน์โหลดมาเล็กผิดปกติ อาจไม่ใช่ไฟล์ .pkl จริง")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error("โหลดโมเดลไม่สำเร็จ")
        st.exception(e)
        st.stop()

model = load_model()

# =========================
# INPUT UI
# =========================
st.header("🔧 ใส่ข้อมูลรถ")

model_name = st.selectbox(
    "รุ่นรถ (model)",
    [
        "1 Series",
        "2 Series",
        "3 Series",
        "4 Series",
        "5 Series",
        "6 Series",
        "7 Series",
        "8 Series",
        "X1",
        "X2",
        "X3",
        "X4",
        "X5",
        "X6",
        "Z4",
        "M2",
        "M3",
        "M4",
        "M5",
        "i3",
        "i8"
    ],
    index=2
)

year = st.number_input("ปีรถ", min_value=1990, max_value=2025, value=2018)
mileage = st.number_input("ระยะทาง (km)", min_value=0, max_value=500000, value=50000)
tax = st.number_input("ภาษี (tax)", min_value=0, max_value=5000, value=150)
mpg = st.number_input("อัตราสิ้นเปลือง (mpg)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
engine_size = st.number_input("ขนาดเครื่องยนต์ (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

fuel_type = st.selectbox(
    "ประเภทเชื้อเพลิง",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)

transmission = st.selectbox(
    "เกียร์",
    ["Automatic", "Manual", "Semi-Auto", "Other"]
)

# =========================
# PREDICT
# =========================
if st.button("💰 ทำนายราคา"):
    try:
        # ต้องตรงกับ feature ตอน train
        input_df = pd.DataFrame({
            "model": [model_name],
            "transmission": [transmission],
            "fuelType": [fuel_type],
            "year": [year],
            "mileage": [mileage],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engine_size]
        })

        prediction = model.predict(input_df)[0]

        st.success(f"💵 ราคาประมาณ: {prediction:,.0f}")

        with st.expander("ดูข้อมูลที่ใช้ทำนาย"):
            st.dataframe(input_df)

    except Exception as e:
        st.error("❌ เกิด error ในการทำนาย")
        st.exception(e)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
