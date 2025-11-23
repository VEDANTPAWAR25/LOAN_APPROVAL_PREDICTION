import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.hybrid_model import predict_proba_from_df
from groq import Groq
from chatbot import loan_chatbot_ui


# ---------------------------------------------------
# SESSION STATE VARIABLES
# ---------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = None

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="wide"
)


# ---------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------
st.markdown("""
<style>
/* Clean compact card */
.card {
    padding: 15px;
    border-radius: 12px;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 12px;
}

/* Predict Button */
.stButton>button {
    width: 100%;
    padding: 12px;
    background: #d62828 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: 0.2s;
}
.stButton>button:hover {
    transform: scale(1.02);
}

/* Move sidebar content slightly up */
.sidebar-title {
    margin-top: -20px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("<h1 style='text-align: center; margin-bottom:8px;'>🏦 Loan Approval Prediction 🏦</h1>", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align:center; margin-top:-5px;'>
Hybrid Model: <b>XGBoost + Deep Neural Network</b>
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; margin-top:-8px; color:#444; font-size:16px;'>
An AI-powered deep learning system that predicts loan approval probability with accuracy and insight.
</p>
""", unsafe_allow_html=True)



# ---------------------------------------------------
# SIDEBAR INPUT FORM
# ---------------------------------------------------
st.sidebar.markdown("<h2 class='sidebar-title'>📋 Applicant Details</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin-top:-10px; margin-bottom:5px;'>", unsafe_allow_html=True)
st.sidebar.write("Provide your information below.")

with st.sidebar.form("input_form"):

    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.subheader("Financial Information")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
    loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360])
    credit_history = st.selectbox("Credit History (1 = good)", [1.0, 0.0], index=0)

    submitted = st.form_submit_button("Predict")



# ---------------------------------------------------
# PREDICTION LOGIC (RUNS ONLY WHEN SUBMITTED)
# ---------------------------------------------------
if submitted:
    total_income = applicant_income + coapplicant_income

    sample = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_emp,
        "ApplicantIncome": int(applicant_income),
        "CoapplicantIncome": int(coapplicant_income),
        "LoanAmount": int(loan_amount),
        "Loan_Amount_Term": int(loan_term),
        "Credit_History": float(credit_history),
        "Property_Area": property_area,
        "Total_Income": int(total_income),
        "Log_Total_Income": np.log1p(total_income),
        "Log_LoanAmount": np.log1p(loan_amount)
    }

    df = pd.DataFrame([sample])
    prob = float(predict_proba_from_df(df)[0])
    label = "Approved" if prob >= 0.5 else "Rejected"

    st.session_state.prediction_result = 1 if label == "Approved" else 0
    st.session_state.user_inputs = sample
    st.session_state.prediction_made = True



# ---------------------------------------------------
# SHOW PREDICTION (ALWAYS PERSISTS)
# ---------------------------------------------------
if st.session_state.prediction_made:

    sample = st.session_state.user_inputs
    prediction_label = "Approved" if st.session_state.prediction_result == 1 else "Rejected"

    df = pd.DataFrame([sample])
    prob = float(predict_proba_from_df(df)[0])

    # ---------------- RESULT CARD ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.metric("Approval Probability", f"{prob*100:.2f}%")

    with col2:
        if prediction_label == "Approved":
            st.success("💚 *Loan Status: APPROVED*")
        else:
            st.error("💔 *Loan Status: REJECTED*")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- INPUT DATA ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Input Data Used for Prediction")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PDF DOWNLOAD ----------------
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Loan Prediction Report")

    c.setFont("Helvetica", 12)
    y = height - 100

    c.drawString(50, y, f"Prediction: {prediction_label}")
    c.drawString(50, y - 25, f"Approval Probability: {prob*100:.2f}%")

    y -= 70
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Input Details:")
    c.setFont("Helvetica", 12)

    y -= 30
    for key, value in sample.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20
        if y < 80:
            c.showPage()
            y = height - 50

    c.save()
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="Loan_Prediction_Report.pdf",
        mime="application/pdf"
    )



# ---------------------------------------------------
# CHATBOT (ALWAYS VISIBLE + KNOWS THE RESULT)
# ---------------------------------------------------
prediction = st.session_state.prediction_result if st.session_state.prediction_made else None
user_inputs = st.session_state.user_inputs if st.session_state.prediction_made else None

st.markdown("<br><br>", unsafe_allow_html=True)
loan_chatbot_ui(prediction_result=prediction, user_inputs=user_inputs)