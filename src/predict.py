# src/predict.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.hybrid_model import predict_proba_from_df

BASE = Path(__file__).resolve().parent.parent
PREPROC_PATH = BASE / "models" / "preprocessor.joblib"

def predict_from_dict(data: dict):
    """data: keys are the column names used in training"""
    df = pd.DataFrame([data])
    probs = predict_proba_from_df(df)
    prob = float(probs[0])
    label = "Approved" if prob >= 0.5 else "Rejected"
    return {"probability": prob, "label": label}

def predict_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    probs = predict_proba_from_df(df)
    labels = ["Approved" if p>=0.5 else "Rejected" for p in probs]
    out = df.copy()
    out["Approved_Prob"] = probs
    out["Prediction"] = labels
    return out

if __name__ == "__main__":
    # example usage
    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 6000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
        "Total_Income": 8000,
        "Log_Total_Income": np.log1p(8000),
        "Log_LoanAmount": np.log1p(120)
    }
    res = predict_from_dict(sample)
    print(res)
