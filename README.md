# Email Spam Detection - Production Machine Learning Web Application

A complete, production-quality supervised machine learning application for classifying email messages into **SPAM** or **NOT SPAM (HAM)**. 

The system implements an end-to-end ML workflow from dataset validation, text preprocessing, TF-IDF feature extraction, and Logistic Regression classification, through model evaluation, pipeline serialization, REST API backend, and a modern responsive user interface.

---

## Architecture & Workflow

```text
Raw Email / Message Input
           ↓
Text Preprocessing & Sanitization
           ↓
TF-IDF Vectorization (Unigrams + Bigrams)
           ↓
Supervised Classifier (Logistic Regression)
           ↓
Predicted Class (SPAM / NOT SPAM) + Confidence Score
```

```text
email_spam_detection/
│
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation & setup guide
│
├── backend/
│   ├── __init__.py               # Python package initialization
│   ├── app.py                    # Flask REST API server & static asset host
│   ├── train_model.py            # Supervised ML training and evaluation script
│   └── spam_emails.csv           # Labelled email dataset
│
├── models/
│   ├── spam_classifier.pkl       # Serialized scikit-learn Pipeline (TF-IDF + Model)
│   └── metrics.json              # Unseen test set evaluation metrics & metadata
│
└── frontend/
    ├── index.html                # Semantic HTML5 web interface
    ├── style.css                 # Custom CSS design system (Dark mode & responsive)
    └── script.js                 # Interactive client logic, API fetch & live metrics
```

---

## Features

- **Trained Supervised ML Pipeline**: Uses `TfidfVectorizer` + `LogisticRegression` for high accuracy, fast inference, and calibrated confidence probabilities.
- **One-Time Training & Reuse**: Trained pipeline is saved to `models/spam_classifier.pkl` and loaded on app start (no retraining on request).
- **Comprehensive Evaluation Metrics**: Evaluated on unseen stratified test data (Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC).
- **RESTful Flask API**: Clean endpoints for `/health`, `/metrics`, and `/predict` with robust request validation, JSON error responses, and CORS support.
- **Modern Responsive Web UI**:
  - Dark mode interface with glassmorphism and subtle glow accents.
  - Interactive quick-sample email picker (Phishing, Lottery, Invoice, Meeting, etc.).
  - Real-time character & word counters with `Ctrl+Enter` shortcut.
  - Distinct alert cards for Spam vs Legitimate messages with confidence gauge and probability breakdown.
  - Live model architecture and test performance metrics loaded directly from the backend.
  - Graceful offline and error states.

---

## Technology Stack

- **Backend**: Python 3, Flask, Flask-CORS
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Model Algorithm**: Logistic Regression (with balanced class weights & L2 regularization)
- **Frontend**: Semantic HTML5, Vanilla Modern CSS, JavaScript (ES6 Fetch API)

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+ (or newer) installed on your system.

### 2. Install Dependencies
In your terminal, navigate to the project root directory and run:

```bash
python -m pip install -r requirements.txt
```

---

## How to Train the Model

To train the machine learning pipeline on the dataset and generate the serialized model and performance metrics, execute:

```bash
python backend/train_model.py
```

### Training Output Summary:
- Automatically detects text and label columns.
- Validates data integrity (cleans missing values, empty strings).
- Performs an 80/20 stratified train/test split.
- Trains the `Pipeline([('tfidf', TfidfVectorizer(...)), ('classifier', LogisticRegression(...))])`.
- Computes evaluation metrics on unseen test data.
- Saves the model to `models/spam_classifier.pkl` and metrics to `models/metrics.json`.

---

## How to Run the Web Application

Start the Flask backend server:

```bash
python backend/app.py
```

Once started, open your web browser and navigate to:

```text
http://127.0.0.1:5000
```

> **Note:** The Flask app serves both the REST API and the frontend web UI directly. If you run the frontend on a separate local development server (e.g. VS Code Live Server on `http://localhost:5500`), CORS is automatically enabled so the frontend connects seamlessly.

---

## REST API Documentation

### 1. Health Check
Checks if the server is running and whether the ML model is loaded in memory.

- **URL**: `GET /health`
- **Response**:
```json
{
  "api_version": "1.0.0",
  "metrics_available": true,
  "model_loaded": true,
  "status": "healthy"
}
```

---

### 2. Model Performance Metrics
Returns the actual evaluation metrics computed during model training on unseen test data.

- **URL**: `GET /metrics`
- **Response**:
```json
{
  "algorithm": "Logistic Regression with L2 Regularization",
  "confusion_matrix": {
    "false_negatives": 0,
    "false_positives": 1,
    "true_negatives": 22,
    "true_positives": 22
  },
  "performance": {
    "accuracy": 97.78,
    "f1_spam": 97.78,
    "precision_spam": 95.65,
    "recall_spam": 100.0,
    "roc_auc": 100.0
  }
}
```

---

### 3. Classify Email (Prediction)
Analyzes an email text string and returns the spam classification verdict and confidence scores.

- **URL**: `POST /predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "email": "URGENT: Your bank account has been suspended. Click here immediately to verify: http://secure-bank-login.xyz/verify"
}
```

- **Success Response (200 OK)**:
```json
{
  "analysis": {
    "character_count": 133,
    "summary": "This email is likely to be spam.",
    "word_count": 17
  },
  "confidence": 99.12,
  "is_spam": true,
  "model_info": {
    "algorithm": "LogisticRegression",
    "vectorizer": "TF-IDF Vectorizer"
  },
  "prediction": "SPAM",
  "probabilities": {
    "ham": 0.0088,
    "spam": 0.9912
  },
  "status": "success"
}
```

- **Validation Error (400 Bad Request)**:
```json
{
  "error": "Email text cannot be empty or contain only whitespace.",
  "status": "error"
}
```

---

## Dataset Schema

The application dataset is located at `backend/spam_emails.csv`.

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `label` | string | Target class: `spam` (unsolicited/phishing) or `ham` (legitimate) |
| `text` | string | Raw email subject and message body content |

---

## Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Model file not found** | `train_model.py` has not been run yet | Run `python backend/train_model.py` to create `models/spam_classifier.pkl` |
| **Backend Offline banner** | Flask server is not running | Run `python backend/app.py` in your terminal |
| **Port 5000 already in use** | Another process is using port 5000 | Kill the competing process or modify the port in `backend/app.py` |
| **ModuleNotFoundError** | Missing Python packages | Run `python -m pip install -r requirements.txt` |
