# src/generate_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_loan_data(rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    genders = ["Male", "Female"]
    married = ["Yes", "No"]
    dependents = ["0", "1", "2", "3+"]
    education = ["Graduate", "Not Graduate"]
    self_employed = ["Yes", "No"]
    prop = ["Rural", "Semiurban", "Urban"]

    # Half approved, half rejected
    half = rows // 2

    # Approved group (Y) - healthier financials
    approved = {
        "Loan_ID": [f"LP_A{i:05d}" for i in range(half)],
        "Gender": np.random.choice(genders, half),
        "Married": np.random.choice(married, half, p=[0.7, 0.3]),
        "Dependents": np.random.choice(dependents, half, p=[0.55, 0.2, 0.15, 0.1]),
        "Education": np.random.choice(education, half, p=[0.75, 0.25]),
        "Self_Employed": np.random.choice(self_employed, half, p=[0.18, 0.82]),
        "ApplicantIncome": np.random.normal(7000, 2500, half).clip(1000, None).astype(int),
        "CoapplicantIncome": np.random.normal(2200, 1200, half).clip(0, None).astype(int),
        "LoanAmount": np.random.normal(140, 40, half).clip(20, None).astype(int),
        "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], half, p=[0.05,0.1,0.2,0.15,0.5]),
        "Credit_History": np.random.choice([1.0, 0.0], half, p=[0.92, 0.08]),
        "Property_Area": np.random.choice(prop, half, p=[0.3, 0.35, 0.35]),
        "Loan_Status": ["Y"] * half
    }

    # Rejected group (N) - weaker financials
    rejected = {
        "Loan_ID": [f"LP_R{i:05d}" for i in range(half)],
        "Gender": np.random.choice(genders, half),
        "Married": np.random.choice(married, half, p=[0.4, 0.6]),
        "Dependents": np.random.choice(dependents, half, p=[0.3, 0.25, 0.25, 0.2]),
        "Education": np.random.choice(education, half, p=[0.5, 0.5]),
        "Self_Employed": np.random.choice(self_employed, half, p=[0.35, 0.65]),
        "ApplicantIncome": np.random.normal(3200, 1600, half).clip(400, None).astype(int),
        "CoapplicantIncome": np.random.normal(800, 900, half).clip(0, None).astype(int),
        "LoanAmount": np.random.normal(200, 70, half).clip(20, None).astype(int),
        "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], half),
        "Credit_History": np.random.choice([1.0, 0.0], half, p=[0.18, 0.82]),
        "Property_Area": np.random.choice(prop, half),
        "Loan_Status": ["N"] * half
    }

    df_a = pd.DataFrame(approved)
    df_r = pd.DataFrame(rejected)
    df = pd.concat([df_a, df_r], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Add derived columns (consistent with preprocess later)
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Log_Total_Income"] = np.log1p(df["Total_Income"])
    df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])

    # Save
    out = Path(__file__).resolve().parent.parent / "data" / "synthetic_loan_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved synthetic dataset to: {out} (rows={len(df)})")
    return df

if __name__ == "__main__":
    generate_synthetic_loan_data(rows=20000)
