import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Get the current directory
BASE_DIR = Path(__file__).parent.parent.absolute()
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'deep_model.h5'
PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.joblib'

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_model_and_preprocessor():
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return preprocessor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and preprocessor
preprocessor, model = load_model_and_preprocessor()

# --- App Title and Description ---
st.title("Loan Approval Prediction 🏦")
st.markdown("""
This application uses a deep neural network to predict loan approval probability 
based on the applicant's details.
""")
st.markdown("---")

# --- Sidebar User Inputs ---
with st.sidebar:
    st.header("📝 Applicant Details")
    st.write("Provide your information below.")
    
    # Personal Information
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    # Financial Information
    st.subheader("Financial Information")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
    loan_term = st.number_input("Loan Term (in months)", 
                               min_value=12, 
                               max_value=480, 
                               value=360,
                               step=12,
                               help="Enter loan duration in months (between 12-480 months)")
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create input dictionary
user_inputs = {
    'Gender': gender,
    'Married': married,
    'Dependents': int(dependents),
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}

# Prediction Section
if model is None or preprocessor is None:
    st.error("⚠️ Model files not found. Please ensure model files are present in the models directory.")
else:
    if st.sidebar.button("Predict Eligibility", type="primary", use_container_width=True):
        try:
            with st.spinner('Analyzing your profile...'):
                # Prepare input data
                input_df = pd.DataFrame([user_inputs])
                input_transformed = preprocessor.transform(input_df)
                
                # Make prediction
                prediction_proba = model.predict(input_transformed, verbose=0)
                prediction = (prediction_proba > 0.5).astype(int)

                # Display results
                st.subheader("🔮 Prediction Result")
                
                if prediction[0][0] == 1:
                    st.success("Congratulations! Your loan application is likely to be **APPROVED**.", icon="🎉")
                    st.balloons()
                    
                    prob_approved = float(prediction_proba[0][0]) * 100
                    st.metric("Approval Confidence", f"{prob_approved:.1f}%")
                    st.progress(prob_approved / 100)
                    
                else:
                    st.error("Unfortunately, your loan application is likely to be **REJECTED**.", icon="😞")
                    
                    prob_rejected = (1 - float(prediction_proba[0][0])) * 100
                    st.metric("Rejection Confidence", f"{prob_rejected:.1f}%")
                    st.progress(prob_rejected / 100)

                # Show details
                with st.expander("View Model Details"):
                    st.code("""
Deep Neural Network Architecture:
- Input Layer
- Dense(128) + BatchNorm + Dropout(0.3)
- Dense(64) + BatchNorm + Dropout(0.2)
- Dense(32) + BatchNorm + Dropout(0.1)
- Output Layer (Sigmoid)
                    """)
                
                with st.expander("View Input Details"):
                    st.dataframe(input_df)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")