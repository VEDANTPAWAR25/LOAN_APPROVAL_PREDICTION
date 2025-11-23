# src/preprocess.py
import pandas as pd
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_loan_data.csv"
PREPROCESS_PATH = Path(__file__).resolve().parent.parent / "models" / "preprocessor.joblib"

def build_and_save_preprocessor(data_path: Path = DATA_PATH, out_path: Path = PREPROCESS_PATH):
    df = pd.read_csv(data_path)

    # Feature lists (must match train and UI)
    numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                        "Loan_Amount_Term", "Credit_History", "Total_Income",
                        "Log_Total_Income", "Log_LoanAmount"]

    categorical_features = ["Gender", "Married", "Dependents", "Education",
                            "Self_Employed", "Property_Area"]

    # Numeric pipeline: impute median + scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: fill missing with most frequent + one-hot
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop",
        sparse_threshold=0
    )

    # fit on whole dataset
    X = df[numeric_features + categorical_features]
    preprocessor.fit(X)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }, out_path)
    print(f"Preprocessor saved to: {out_path}")
    return preprocessor

if __name__ == "__main__":
    build_and_save_preprocessor()
