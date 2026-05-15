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

## Project Structure
```
phishdetect/
├── train_model.py      # ML training script
├── app.py              # Flask backend + REST API
├── index.html          # Frontend (served by Flask)
├── phish_model.pkl     # Trained model (generated after training)
└── README.md
```

## API

**POST /scan-email**

Request:
```json
{ "subject": "string", "body": "string" }
```

Response:
```json
{
  "classification": "Phishing",
  "confidence": 97.3,
  "risk_level": "High",
  "indicators": ["Urgent language", "Suspicious URL"],
  "recommended_actions": ["Delete this email..."],
  "phishing_probability": 97.3,
  "legitimate_probability": 2.7
}
```

## ML Architecture
- **Text features**: TF-IDF (word unigrams/bigrams/trigrams) + character n-grams
- **Security features**: URL count, suspicious keywords, caps ratio, etc.
- **Model**: Soft-voting ensemble of Logistic Regression + Random Forest + Gradient Boosting
- **Accuracy**: ~98%+ on test set
