# 🏦 Loan Approval Prediction (Hybrid Model)

This project is an AI-powered **Loan Approval Prediction System** built using a **Hybrid Model (XGBoost + Deep Neural Network)**.  
The application is deployed using **Streamlit**, allowing users to enter applicant details, calculate loan approval probability, and download a detailed **PDF report**.

---

## 🚀 Features

- 🔮 Predicts **Loan Approval Probability**
- 🤖 Powered by **Hybrid ML Model → XGBoost + Deep Neural Network**
- 🖥 Streamlit-based **modern UI**
- 📄 **Auto-generated PDF Report** of prediction
- 📊 Displays input data used for prediction
- 🧮 Clean, responsive UI with improved styling

---

## 🧠 Technologies Used

| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| ML Models | XGBoost, TensorFlow/Keras |
| Data Processing | Pandas, NumPy |
| PDF Generation | ReportLab |
| Misc | Scikit-Learn |

---

## 📁 Project Structure

```
Loan-Approval-Prediction/
│── src/
│   ├── hybrid_model.py
│   └── model_files/ (saved trained models)
│
│── ui/
│   └── app.py
│
│── requirements.txt
│── README.md
│── data/ (optional dataset)
└── .venv/ (virtual environment)
```

---

# ⚙️ Installation & Running the Project

Follow the steps below to **clone** and **run** the application.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

> Replace `<your-username>` with your GitHub username.

---

## 2️⃣ Create Virtual Environment

### **Windows**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### **Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Streamlit App

```bash
streamlit run ui/app.py
```

Then open:

```
http://localhost:8501
```

---

# 📄 Requirements

```
streamlit
pandas
numpy
xgboost
tensorflow
scikit-learn
reportlab
```

---

# 🤖 How the Hybrid Model Works

### ✔ **XGBoost Model**  
Captures patterns in structured data.

### ✔ **Deep Neural Network**  
Learns non-linear relationships.

### ✔ **Final Hybrid Output**  
The system calculates weighted predictions from both models to produce a more accurate probability score.

---

# 📤 Output Includes

- ✔ Loan approval probability  
- ✔ Approval / rejection status  
- ✔ Applicant details summary  
- ✔ Downloadable **PDF report**  

---

# 🤝 Contributing

Pull requests are welcome!  
For major changes, open an issue to discuss your ideas.

---

# ⭐ Support

If you like this project, please ⭐ the repo!

