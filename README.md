Problem Definition

ตลาดรถมือสองมีราคาที่แตกต่างกันตามหลายปัจจัย เช่น รุ่นรถ ปี ระยะทาง และสเปคของรถ  
ทำให้ผู้ซื้อและผู้ขายประเมินราคาที่เหมาะสมได้ยาก

โปรเจกต์นี้มีเป้าหมายเพื่อ:
- ช่วยประเมินราคากลางของรถ BMW มือสอง
- ลดความไม่แน่นอนในการตั้งราคาซื้อขาย
- ใช้ข้อมูลจริงมาช่วยตัดสินใจ

---

Dataset

Dataset ถูกนำมาจาก:
👉 Kaggle: BMW Used Car Dataset

ซึ่งเป็น dataset ที่รวบรวมข้อมูลรถ BMW มือสองจากตลาดจริง  
จึงมีความน่าเชื่อถือและเหมาะกับการนำมาวิเคราะห์ราคา

Problem Type
Regression (ทำนายราคา)

Feature Description 

- **Model** → รุ่นของรถ (มีผลต่อราคาโดยตรง)
- **Transmission** → ประเภทเกียร์ (Auto มักแพงกว่า)
- **Fuel Type** → ประเภทเชื้อเพลิง
- **Year** → ปีของรถ (รถใหม่ราคาสูงกว่า)
- **Mileage** → ระยะทางใช้งาน (ยิ่งเยอะราคายิ่งลด)
- **Tax** → ภาษีรถ (สะท้อนต้นทุน)
- **MPG** → ความประหยัดน้ำมัน
- **Engine Size** → ขนาดเครื่องยนต์ (แรง = แพง)

🔹 Algorithms Tested
- RandomForestRegressor  (เลือกใช้)
- ExtraTreesRegressor
- GradientBoostingRegressor

🔹 Hyperparameter Tuning
ใช้ **RandomizedSearchCV + Cross Validation**

🔹 Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Performance

| Metric | Value |
|------|------|
| CV MAE Mean | 1536.45 |
| CV MAE Std | 28.83 |
| CV R² Mean | 0.9458 |
| Test MAE | 1606.54 |
| Test RMSE | 2745.23 |
| Test R² | 0.9417 |

โมเดลมีความแม่นยำสูง และไม่มีปัญหา overfitting

### พัฒนาโดยใช้ Streamlit

ฟีเจอร์:
- กรอกข้อมูลรถผ่าน UI
- ทำนายราคาทันที
- แสดงช่วงราคาประมาณ
- แสดงข้อมูล model performance
- มี validation และ disclaimer

Why Machine Learning?

ปัญหานี้ไม่สามารถใช้สูตรตรงๆ ได้ เพราะ:
- ความสัมพันธ์ของข้อมูลเป็น nonlinear
- มีหลาย feature ที่ส่งผลร่วมกัน

Machine Learning จึงเหมาะสำหรับ:
- เรียนรู้ pattern จากข้อมูลจริง
- ทำนายราคาได้แม่นยำกว่าวิธี manual