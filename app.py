import os
import joblib
import gdown
import pandas as pd
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 900px;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #9ca3af;
    margin-bottom: 1.2rem;
}
.section-card {
    background: #111827;
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.result-card {
    background: linear-gradient(135deg, #0f172a, #14532d);
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 1rem;
}
.small-label {
    font-size: 0.95rem;
    color: #cbd5e1;
    margin-bottom: 0.3rem;
}
.big-price {
    font-size: 2rem;
    font-weight: 800;
    color: #86efac;
}
.footer-text {
    text-align: center;
    color: #9ca3af;
    font-size: 0.9rem;
    margin-top: 2rem;
}
.hero-img {
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="main-title">🚗 BMW Used Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ทำนายราคารถมือสอง BMW ด้วย Machine Learning</div>', unsafe_allow_html=True)

# =========================
# TOP IMAGE
# =========================
image_path = "assets/bmw_m4.jpg"
if os.path.exists(image_path):
    st.image(image_path, caption="BMW M4", use_container_width=True)
else:
    st.info("ยังไม่พบรูป BMW M4 กรุณาใส่ไฟล์รูปไว้ที่ assets/bmw_m4.jpg")

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
        with st.spinner("กำลังดาวน์โหลดโมเดลครั้งแรก..."):
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
        st.error("ไฟล์โมเดลที่ดาวน์โหลดมาไม่ถูกต้อง")
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
# INPUT FORM
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔧 ใส่ข้อมูลรถ")

col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox(
        "รุ่นรถ (Model)",
        [
            "1 Series", "2 Series", "3 Series", "4 Series", "5 Series",
            "6 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4",
            "X5", "X6", "Z4", "M2", "M3", "M4", "M5", "i3", "i8"
        ],
        index=18
    )

    year = st.number_input("ปีรถ (Year)", min_value=1990, max_value=2025, value=2018)
    mileage = st.number_input("ระยะทาง (Mileage: km)", min_value=0, max_value=500000, value=50000)
    engine_size = st.number_input("ขนาดเครื่องยนต์ (Engine Size: L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

with col2:
    tax = st.number_input("ภาษีรถ (Tax)", min_value=0, max_value=5000, value=150)
    mpg = st.number_input("อัตราสิ้นเปลือง (MPG)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)

    fuel_type = st.selectbox(
        "ประเภทเชื้อเพลิง (Fuel Type)",
        ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
    )

    transmission = st.selectbox(
        "ระบบเกียร์ (Transmission)",
        ["Automatic", "Manual", "Semi-Auto", "Other"]
    )

predict_btn = st.button("💰 ทำนายราคา (Predict Price)", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICT
# =========================
if predict_btn:
    try:
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

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">ราคาประเมิน (Estimated Price)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-price">${prediction:,.0f}</div>', unsafe_allow_html=True)
        st.caption("สกุลเงิน: USD")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("ดูข้อมูลที่ใช้ในการทำนาย"):
            show_df = input_df.copy()
            show_df.columns = [
                "รุ่นรถ", "ระบบเกียร์", "ประเภทเชื้อเพลิง", "ปีรถ",
                "ระยะทาง", "ภาษี", "MPG", "ขนาดเครื่องยนต์"
            ]
            st.dataframe(show_df, use_container_width=True)

    except Exception as e:
        st.error("เกิดข้อผิดพลาดในการทำนายราคา")
        st.exception(e)

# =========================
# FOOTER
# =========================
st.markdown('<div class="footer-text">Developed with ❤️ using Streamlit</div>', unsafe_allow_html=True)
