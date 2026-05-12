# PhishDetect — Phishing Email Detection System

## Prerequisites
- Python 3.8+
- pip packages: `flask scikit-learn numpy pandas`

Install dependencies:
```bash
pip install flask scikit-learn numpy pandas
```

## Setup & Run

### Step 1 — Train the ML model (one-time)
```bash
python train_model.py
```
This trains the ensemble model (Logistic Regression + Random Forest + Gradient Boosting)
and saves it as `phish_model.pkl`.

### Step 2 — Start the server
```bash
python app.py
```
Server starts at: **http://localhost:5000**

### Step 3 — Open in browser
Navigate to `http://localhost:5000`

---

