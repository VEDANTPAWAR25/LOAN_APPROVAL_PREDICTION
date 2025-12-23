# 🏦 Loan Approval Prediction (Hybrid Model)

This project is an AI-powered **Loan Approval Prediction System** built using a **Hybrid Deep Learning Model (XGBoost + Deep Neural Network)**. The application is deployed using **Streamlit**, allowing users to enter applicant details, calculate loan approval probability, and download detailed **PDF reports**. It also features an **AI-powered chatbot** (powered by Groq) for loan guidance.

---

## 🚀 Features

- 🔮 **Predicts Loan Approval Probability** with high accuracy
- 🤖 **Hybrid ML Model**: XGBoost + Deep Neural Network ensemble
- 🖥️ **Streamlit-based Modern UI** with responsive design
- 📄 **Auto-generated PDF Reports** with prediction details
- 📊 **Input Data Visualization** showing features used for prediction
- 💬 **AI-Powered Chatbot** (Groq LLaMA 3.1) for loan assistance
- 🎨 **Clean, Glassmorphic UI** with improved styling
- 📱 **Sidebar Form** for easy data input

---

## 🧠 Technologies Used

| Component | Technology |
|----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML Models** | XGBoost, TensorFlow/Keras |
| **Data Processing** | Pandas, NumPy, Scikit-Learn |
| **PDF Generation** | ReportLab |
| **AI Chatbot** | Groq (LLaMA 3.1-8b) |
| **Feature Engineering** | Scikit-Learn (ColumnTransformer, StandardScaler, OneHotEncoder) |

---

## 📁 Project Structure

```
Loan-Prediction-DL/
│
├── 📂 src/
│   ├── __init__.py
│   ├── generate_dataset.py       # Synthetic data generation (20,000 records)
│   ├── preprocess.py             # Feature engineering & preprocessing pipeline
│   ├── train_model.py            # Model training (XGBoost + DNN)
│   ├── hybrid_model.py           # Inference & prediction logic
│   ├── predict.py                # Prediction utilities
│   │
│   └── 📂 config/
│       ├── __init__.py
│       └── groq_key.py           # Groq API key configuration
│
├── 📂 ui/
│   ├── app.py                    # Main Streamlit application
│   └── chatbot.py                # Chatbot UI component
│
├── 📂 models/
│   ├── dnn_model.h5              # Trained DNN model (Keras)
│   └── preprocessor.joblib       # Saved preprocessor (features + scalers)
│
├── 📂 data/
│   └── synthetic_loan_data.csv   # Generated dataset (20,000 rows)
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

---

## 🧪 Dataset Overview

The project uses **synthetically generated loan data** with the following features:

### **Categorical Features:**
- Gender (Male, Female)
- Married (Yes, No)
- Dependents (0, 1, 2, 3+)
- Education (Graduate, Not Graduate)
- Self_Employed (Yes, No)
- Property_Area (Rural, Semiurban, Urban)

### **Numerical Features:**
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History (1.0 = good, 0.0 = bad)

### **Derived Features:**
- Total_Income (ApplicantIncome + CoapplicantIncome)
- Log_Total_Income (log transformation)
- Log_LoanAmount (log transformation)

### **Target Variable:**
- Loan_Status (Y = Approved, N = Rejected)

**Dataset Size**: 20,000 records (50% approved, 50% rejected)

---

## 🤖 How the Hybrid Model Works

### **1️⃣ Data Preprocessing Pipeline**

Located in [`src/preprocess.py`](src/preprocess.py):

- **Numeric Features**: Imputation (median) + StandardScaler normalization
- **Categorical Features**: Imputation (most frequent) + OneHotEncoder
- **Pipeline**: Scikit-Learn ColumnTransformer for unified preprocessing

### **2️⃣ XGBoost Model**

Located in [`src/train_model.py`](src/train_model.py):

- **Algorithm**: XGBoost Classifier
- **Configuration**:
  - 500 estimators, max_depth=6, learning_rate=0.05
  - Captures non-linear patterns in structured financial data
  - Outputs probability scores for each sample

### **3️⃣ Deep Neural Network (DNN)**

Architecture:
```
Input Layer (n_features + 1)
    ↓
Dense(256) → BatchNorm → Dropout(0.3)
    ↓
Dense(128) → BatchNorm → Dropout(0.2)
    ↓
Dense(64) → ReLU
    ↓
Dense(1) → Sigmoid (probability output)
```

**Training Details**:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 12, Batch Size: 128
- Validation Split: 12%
- Dropout regularization to prevent overfitting

### **4️⃣ Hybrid Ensemble**

Located in [`src/hybrid_model.py`](src/hybrid_model.py):

The final prediction is a **weighted ensemble**:
```
Final_Probability = 0.6 × XGBoost_Prob + 0.4 × DNN_Prob
```

This combination leverages:
- **XGBoost** (60%): Fast, accurate tree-based learning
- **DNN** (40%): Deep non-linear feature interactions

---

## 📊 Model Performance

The hybrid model is evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Train-Test Split**: 85% training, 15% testing (stratified)

---

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+
- pip package manager

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/<your-username>/Loan-Prediction-DL.git
cd Loan-Prediction-DL
```

### **2️⃣ Create Virtual Environment**

**Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4️⃣ Setup Groq API Key**

1. Get your Groq API key from [console.groq.com](https://console.groq.com)
2. Update [`src/config/groq_key.py`](src/config/groq_key.py):

```python
GROQ_API_KEY = "your-api-key-here"
```

### **5️⃣ Generate Dataset & Train Model**

```bash
# Generate synthetic dataset
python src/generate_dataset.py

# Train the hybrid model
python src/train_model.py
```

### **6️⃣ Run the Streamlit App**

```bash
streamlit run ui/app.py
```

Then open your browser to:
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit==1.31.0
pandas==2.0.3
numpy==1.24.3
xgboost==2.0.0
tensorflow==2.13.0
scikit-learn==1.3.0
reportlab==4.0.4
groq==0.4.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### **Via Streamlit UI** (Recommended)

1. Run `streamlit run ui/app.py`
2. Fill in applicant details in the sidebar form
3. Click **"Predict"** button
4. View prediction results with:
   - Approval probability percentage
   - Approval/Rejection status
   - Input data summary
   - Download PDF report
5. Chat with the AI assistant for loan guidance

### **Via Python API**

[`src/predict.py`](src/predict.py) provides programmatic access:

```python
from src.predict import predict_from_dict

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
    "Log_Total_Income": 8.98,
    "Log_LoanAmount": 4.79
}

result = predict_from_dict(sample)
print(result)  # {'probability': 0.87, 'label': 'Approved'}
```

---

## 🤖 AI Chatbot Features

The chatbot (powered by **Groq LLaMA 3.1-8b**) provides:

- 💬 Real-time responses to loan-related queries
- 📚 Context-aware guidance based on your prediction result
- 🎯 Loan improvement tips
- ❓ General loan assistance

Located in [`ui/chatbot.py`](ui/chatbot.py) and integrated into [`ui/app.py`](ui/app.py).

---

## 📤 Outputs

The application provides:

✅ **Loan Approval Probability** (0-100%)  
✅ **Approval/Rejection Status** (binary decision at 50% threshold)  
✅ **Applicant Details Summary** (all input features)  
✅ **Downloadable PDF Report** with prediction details  
✅ **AI Chatbot Assistance** for loan-related questions  

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| [`src/generate_dataset.py`](src/generate_dataset.py) | Generates 20,000 synthetic loan records |
| [`src/preprocess.py`](src/preprocess.py) | Feature engineering & preprocessing pipeline |
| [`src/train_model.py`](src/train_model.py) | Trains XGBoost + DNN hybrid model |
| [`src/hybrid_model.py`](src/hybrid_model.py) | Model inference & prediction logic |
| [`src/predict.py`](src/predict.py) | High-level prediction API |
| [`ui/app.py`](ui/app.py) | Main Streamlit application |
| [`ui/chatbot.py`](ui/chatbot.py) | AI chatbot component |
| [`src/config/groq_key.py`](src/config/groq_key.py) | Groq API key configuration |

---

## 🧪 Testing

To test the prediction pipeline:

```bash
python src/predict.py
```

This runs the example in [`src/predict.py`](src/predict.py) and prints the prediction result.

---

## 🎯 Future Enhancements

- [ ] Model explainability (SHAP values)
- [ ] Hyperparameter tuning with Optuna
- [ ] Model versioning & experiment tracking
- [ ] Database integration for storing predictions
- [ ] API deployment (FastAPI/Flask)
- [ ] Additional ensemble methods (LightGBM, CatBoost)
- [ ] Feature importance visualization
- [ ] Real-world dataset integration

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## ⭐ Support

If you find this project helpful, please **⭐ star** the repository!

For issues or questions, feel free to **open an issue** on GitHub.

---

## 👨‍💻 Author

**Vedant** - AI/ML Enthusiast  
GitHub: [@VEDANTPAWAR25](https://github.com/VEDANTPAWAR25/LOAN_APPROVAL_PREDICTION.git)

---

## 🙏 Acknowledgments

- **XGBoost** team for the powerful gradient boosting library
- **TensorFlow/Keras** for deep learning framework
- **Streamlit** for the beautiful UI framework
- **Groq** for the fast LLM inference API
- **Scikit-Learn** for preprocessing & utilities

---


**Project Status**: ✅ Active & Maintained
