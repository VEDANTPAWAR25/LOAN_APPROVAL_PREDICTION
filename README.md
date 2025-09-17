# Loan Prediction System

A deep learning-based system that predicts loan approval status using neural networks.

## Project Overview

This project implements a machine learning solution for predicting loan approval status based on applicant information. It uses a deep neural network built with TensorFlow to analyze various applicant features and determine loan eligibility.

## Technical Architecture

### Model Specifications
- **Framework**: TensorFlow/Keras
- **Architecture**: Multi-layer Neural Network
  - Input Layer: Feature-dimensioned
  - Hidden Layers: Dense (64 → 32 → 16 nodes)
  - Output Layer: Single node with sigmoid activation
  - Loss Function: Binary Cross-entropy
  - Optimizer: Adam

### Features Used
- **Categorical Features**:
  - Gender
  - Marital Status
  - Education
  - Self Employment
  - Property Area
- **Numerical Features**:
  - Applicant Income
  - Co-applicant Income
  - Loan Amount
  - Loan Term
  - Credit History
  - Dependencies

## Setup and Usage

1. Clone the repository:
```bash
git clone https://github.com/VEDANTPAWAR25/LOAN_APPROVAL_PREDICTION.git
cd LOAN_APPROVAL_PREDICTION
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train_and_save_model.py
```

4. Run the web interface:
```bash
streamlit run src/app.py
```

## Project Structure
```
├── data/
│   └── train.csv
├── models/
│   ├── deep_model.h5
│   └── preprocessor.joblib
├── src/
│   ├── train_and_save_model.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Model Performance
- Test Accuracy: ~76%
- Validation Split: 20%
- Batch Size: 32
- Epochs: 100

## Technologies Used
- Python 3.x
- TensorFlow 2.x
- Streamlit
- Pandas
- Scikit-learn
- NumPy