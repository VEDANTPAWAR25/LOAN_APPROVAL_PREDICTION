# src/hybrid_model.py
import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf

BASE = Path(__file__).resolve().parent.parent
PREPROC_PATH = BASE / "models" / "preprocessor.joblib"
XGB_OUT = BASE / "models" / "xgb_model.pkl"
DNN_OUT = BASE / "models" / "dnn_model.h5"

def load_artifacts():
    meta = joblib.load(PREPROC_PATH)
    preprocessor = meta["preprocessor"]
    numeric_features = meta["numeric_features"]
    categorical_features = meta["categorical_features"]

    xgb_model = joblib.load(XGB_OUT)
    dnn_model = tf.keras.models.load_model(DNN_OUT)
    return preprocessor, numeric_features, categorical_features, xgb_model, dnn_model

def predict_proba_from_df(df):
    preprocessor, numeric_features, categorical_features, xgb_model, dnn_model = load_artifacts()
    X_raw = df[numeric_features + categorical_features]
    X_trans = preprocessor.transform(X_raw)
    try:
        X_arr = X_trans.toarray()
    except Exception:
        X_arr = np.asarray(X_trans)
    xgb_prob = xgb_model.predict_proba(X_arr)[:, 1].reshape(-1, 1)
    hybrid_input = np.hstack([X_arr, xgb_prob])
    dnn_prob = dnn_model.predict(hybrid_input).reshape(-1)
    # weighted ensemble: 0.6 * xgb + 0.4 * dnn
    final = 0.6 * xgb_prob.reshape(-1) + 0.4 * dnn_prob
    return final
