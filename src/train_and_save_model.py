import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from keras import models, layers
import joblib
from pathlib import Path

# Configure TensorFlow
tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()

# Constants
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
DATA_PATH = DATA_DIR / 'train.csv'
MODEL_PATH = MODEL_DIR / 'deep_model.h5'
PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.joblib'

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("Loading and preprocessing data...")
    
    df = pd.read_csv(DATA_PATH)
    df = df.drop("Loan_ID", axis=1)
    
    # Handle missing values
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area']
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                     'Loan_Amount_Term', 'Credit_History']
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical values
    df["Dependents"] = df["Dependents"].replace('3+', '3').astype(float)
    df["Loan_Status"] = df["Loan_Status"].map({'Y': 1, 'N': 0})
    
    return df

def create_preprocessor(categorical_features, numerical_features):
    """Create and return the preprocessor."""
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def build_model(input_shape):
    """Build and return the deep learning model."""
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Split features and target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define features
    categorical_features = ['Gender', 'Married', 'Education', 
                          'Self_Employed', 'Property_Area']
    numerical_features = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome',
                        'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor(categorical_features, numerical_features)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("Preprocessor saved successfully!")
    
    # Build and train model
    print("\nBuilding and training deep learning model...")
    model = build_model(X_train_transformed.shape[1])
    
    history = model.fit(
        X_train_transformed, 
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    results = model.evaluate(X_test_transformed, y_test, verbose=1)
    print(f"\nTest accuracy: {results[1]:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()