import streamlit as st
import requests
import pytesseract
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Loan AI System", layout="centered")

st.title("🏦 AI Loan Risk Analyzer")

st.write("Enter customer details:")

# ==============================
# OCR FUNCTION
# ==============================
def extract_data_from_image(image):
    text = pytesseract.image_to_string(image)

    st.write("### 📄 Extracted Text")
    st.code(text)

    def extract(pattern, default=0):
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else default

    data = {
        "ApplicantIncome": extract(r'Income[:\s]+(\d+)'),
        "CoapplicantIncome": 0,
        "LoanAmount": extract(r'Loan Amount[:\s]+(\d+)'),
        "Loan_Amount_Term": 360,
        "Credit_History": 1 if "good" in text.lower() else 0,
        "Gender": 1 if "male" in text.lower() else 0,
        "Married": 1 if "married" in text.lower() else 0,
        "Education": 1 if "graduate" in text.lower() else 0,
        "Self_Employed": 1 if "self employed" in text.lower() else 0,
        "Property_Area": 2 if "urban" in text.lower() else 1
    }

    return data


uploaded_file = st.file_uploader("📤 Upload Loan Document", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Document", use_column_width=True)

    ocr_data = extract_data_from_image(image)

    st.write("### 🧾 Auto Extracted Data")
    st.json(ocr_data)
# ==============================
# INPUTS
# ==============================

income = st.number_input("Applicant Income", 0, 100000, 5000)
co_income = st.number_input("Coapplicant Income", 0, 100000, 0)
loan = st.number_input("Loan Amount", 0, 1000, 150)
term = st.number_input("Loan Term", 0, 500, 360)

credit = st.selectbox("Credit History", [1, 0])
gender = st.selectbox("Gender", [1, 0])
married = st.selectbox("Married", [1, 0])

education = st.selectbox("Education", [1, 0])
self_emp = st.selectbox("Self Employed", [1, 0])
property_area = st.selectbox("Property Area", [0, 1, 2])

# ==============================
# PREDICT BUTTON
# ==============================


if st.button("Predict"):

    data = {
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan,
        "Loan_Amount_Term": term,
        "Credit_History": credit,
        "Gender": gender,
        "Married": married,
        "Education": education,
        "Self_Employed": self_emp,
        "Property_Area": property_area
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)

        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("📊 Result")

            if "Reject" in result["decision"]:
                st.error(result["decision"])
            else:
                st.success(result["decision"])

            st.write("### 🔢 Risk Probability")
            st.write(f"{result['default_probability_percent']}%")

            st.write("### ⚠️ Risk Level")
            st.write(result["risk_level"])

            st.write("### 📌 Top Factors")
            st.write(result["top_factors"])

            st.write("### 🤖 AI Explanation")
            st.info(result["ai_explanation"])

    except Exception as e:
        st.error(f"API Error: {e}")