"""
Email Spam Detection - Flask Backend Application
------------------------------------------------
Provides REST API endpoints for real-time supervised spam classification,
health checks, model performance metrics, and static frontend hosting.
"""

import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

# Resolve project directories
BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "spam_classifier.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

# Initialize Flask application
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)  # Enable Cross-Origin Resource Sharing for external frontend dev servers

# Maximum allowed email input length (characters)
MAX_EMAIL_LENGTH = 50000

# Global model and metrics holders
model_pipeline = None
model_metrics = None


def load_saved_model():
    """Load the trained machine learning pipeline and metrics."""
    global model_pipeline, model_metrics
    
    if MODEL_FILE.exists():
        try:
            model_pipeline = joblib.load(MODEL_FILE)
            print(f"[+] Loaded ML pipeline from {MODEL_FILE}")
        except Exception as e:
            print(f"[!] Error loading model file: {e}", file=sys.stderr)
            model_pipeline = None
    else:
        print(
            f"\n[!] WARNING: Model file not found at '{MODEL_FILE}'.\n"
            f"    Please run the training script first:\n"
            f"    python backend/train_model.py\n",
            file=sys.stderr
        )
        model_pipeline = None
        
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
            print(f"[+] Loaded evaluation metrics from {METRICS_FILE}")
        except Exception as e:
            print(f"[!] Error loading metrics file: {e}", file=sys.stderr)
            model_metrics = None
    else:
        model_metrics = None


# Load model upon module initialization
load_saved_model()


@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve frontend index.html."""
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({
        "message": "Email Spam Detection API is running.",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "metrics": "/metrics"
        }
    })


@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    """Serve static frontend assets (css, js, images)."""
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / path).exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return jsonify({"error": "Resource not found"}), 404


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Returns status of the server, model availability, and metrics.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "metrics_available": model_metrics is not None,
        "api_version": "1.0.0"
    }), 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Returns actual evaluation metrics generated during training.
    """
    if model_metrics is not None:
        return jsonify(model_metrics), 200
        
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return jsonify(metrics), 200
        except Exception as e:
            return jsonify({"error": f"Failed to load metrics: {str(e)}"}), 500
            
    return jsonify({
        "error": "Model metrics are unavailable. Train the model first.",
        "instruction": "Run 'python backend/train_model.py' to generate metrics."
    }), 404


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether an email message is SPAM or NOT SPAM (HAM).
    Expected JSON payload:
    {
        "email": "Your email text content here..."
    }
    """
    # 1. Verify model is loaded
    if model_pipeline is None:
        # Attempt one reload in case training completed while server was running
        load_saved_model()
        if model_pipeline is None:
            return jsonify({
                "status": "error",
                "error": "Model file not found. Please run 'python backend/train_model.py' to train the model."
            }), 503

    # 2. Validate Request JSON
    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": "Request Content-Type must be 'application/json'."
        }), 400
        
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "error": "Malformed JSON payload. Expected a JSON object."
        }), 400
        
    # 3. Extract & Validate Email Text
    # Accept 'email', 'text', or 'message' keys for flexibility
    email_text = data.get("email") or data.get("text") or data.get("message")
    
    if email_text is None:
        return jsonify({
            "status": "error",
            "error": "Missing required field 'email' in JSON request."
        }), 400
        
    if not isinstance(email_text, str):
        return jsonify({
            "status": "error",
            "error": "Field 'email' must be a string."
        }), 400
        
    cleaned_text = email_text.strip()
    if not cleaned_text:
        return jsonify({
            "status": "error",
            "error": "Email text cannot be empty or contain only whitespace."
        }), 400
        
    if len(email_text) > MAX_EMAIL_LENGTH:
        return jsonify({
            "status": "error",
            "error": f"Email text exceeds maximum allowed limit of {MAX_EMAIL_LENGTH} characters."
        }), 400
        
    try:
        # 4. Perform Supervised Prediction via Saved Pipeline
        raw_pred = model_pipeline.predict([cleaned_text])[0]
        
        # Determine probabilities if model supports predict_proba
        probabilities = {}
        confidence = 0.0
        
        if hasattr(model_pipeline, "predict_proba"):
            proba_arr = model_pipeline.predict_proba([cleaned_text])[0]
            classes = list(model_pipeline.classes_)
            
            for cls_name, prob in zip(classes, proba_arr):
                probabilities[str(cls_name)] = round(float(prob), 4)
                
            max_prob = float(max(proba_arr))
            confidence = round(max_prob * 100, 2)
        
        is_spam = str(raw_pred).lower() == "spam"
        prediction_label = "SPAM" if is_spam else "NOT SPAM"
        
        # Word and character metrics
        char_count = len(cleaned_text)
        word_count = len(cleaned_text.split())
        
        # Get model metadata
        clf_step = model_pipeline.named_steps.get("classifier")
        algorithm_name = type(clf_step).__name__ if clf_step is not None else "Supervised Classifier"
        
        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "is_spam": is_spam,
            "confidence": confidence,
            "probabilities": {
                "spam": probabilities.get("spam", 0.0),
                "ham": probabilities.get("ham", 0.0)
            },
            "analysis": {
                "character_count": char_count,
                "word_count": word_count,
                "summary": "This email is likely to be spam." if is_spam else "This email appears to be legitimate."
            },
            "model_info": {
                "algorithm": algorithm_name,
                "vectorizer": "TF-IDF Vectorizer"
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"An error occurred during prediction: {str(e)}"
        }), 500


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "error", "error": "Bad request", "details": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"status": "error", "error": "HTTP method not allowed for this endpoint"}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "error": "Internal server error"}), 500


def main():
    """Start the Flask application server."""
    port = 5000
    host = "127.0.0.1"
    print(f"==================================================")
    print(f"       EMAIL SPAM DETECTION - WEB BACKEND         ")
    print(f"==================================================")
    print(f"[*] Starting server on http://{host}:{port}")
    print(f"[*] Frontend accessible at: http://{host}:{port}/")
    print(f"[*] API Health check at   : http://{host}:{port}/health")
    print(f"[*] API Predict endpoint  : POST http://{host}:{port}/predict")
    print(f"[*] Model Metrics endpoint: GET http://{host}:{port}/metrics")
    print(f"==================================================\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
