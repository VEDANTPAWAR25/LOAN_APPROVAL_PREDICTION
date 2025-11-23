# src/train_model.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import tensorflow as tf

from src.preprocess import build_and_save_preprocessor

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "synthetic_loan_data.csv"
PREPROC_PATH = BASE / "models" / "preprocessor.joblib"
XGB_OUT = BASE / "models" / "xgb_model.pkl"
DNN_OUT = BASE / "models" / "dnn_model.h5"

def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df["Loan_Status"].map({"N": 0, "Y": 1}).astype(int)
    return df, y

def train():
    # ensure preprocessor exists and load it
    preproc = build_and_save_preprocessor(DATA_PATH, PREPROC_PATH)
    meta = joblib.load(PREPROC_PATH)
    preprocessor = meta["preprocessor"]

    df, y = load_data()
    X_raw = df[meta["numeric_features"] + meta["categorical_features"]]

    # transform to numpy array
    X_all = preprocessor.transform(X_raw)
    # if it's sparse, convert to array
    try:
        X_all = X_all.toarray()
    except Exception:
        X_all = np.asarray(X_all)

    # Train-test split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.15, random_state=42, stratify=y
    )

    # ---------- XGBoost ----------
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        use_label_encoder=False,
        eval_metric="logloss",
        tree_method="hist"
    )
    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)

    # predict probabilities on train and test
    train_probs = xgb_model.predict_proba(X_train)[:, 1]
    test_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Save XGBoost
    joblib.dump(xgb_model, XGB_OUT)
    print(f"Saved XGBoost to {XGB_OUT}")

    # ---------- DNN (takes original features + XGB prob) ----------
    X_train_h = np.hstack([X_train, train_probs.reshape(-1, 1)])
    X_test_h = np.hstack([X_test, test_probs.reshape(-1, 1)])

    input_dim = X_train_h.shape[1]
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Training DNN hybrid model...")
    model.fit(X_train_h, y_train, validation_split=0.12, epochs=12, batch_size=128, verbose=2)

    # Evaluate
    preds = (model.predict(X_test_h) > 0.5).astype(int).reshape(-1)
    acc = accuracy_score(y_test, preds)
    print("Hybrid DNN accuracy:", acc)
    print(classification_report(y_test, preds))

    # Save DNN
    model.save(DNN_OUT)
    print(f"Saved DNN model to {DNN_OUT}")

    # Save preprocessor (already saved), but copy again for safety
    joblib.dump(meta, PREPROC_PATH)
    print("Training complete.")
    return xgb_model, model

if __name__ == "__main__":
    train()
