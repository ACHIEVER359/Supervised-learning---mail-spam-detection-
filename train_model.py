"""
Email Spam Detection - Machine Learning Training Script
------------------------------------------------------
This script loads, validates, preprocesses the email spam dataset,
builds a supervised TF-IDF + Logistic Regression classification pipeline,
evaluates the model on unseen test data, and saves the trained pipeline
and evaluation metrics for reuse by the backend application.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib

# Set reproducible random seed
RANDOM_STATE = 42


def get_project_paths():
    """Resolve project directories path-agnostically."""
    backend_dir = Path(__file__).resolve().parent
    base_dir = backend_dir.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Possible dataset locations in order of priority
    possible_dataset_paths = [
        backend_dir / "spam_emails.csv",
        base_dir / "spam_emails.csv",
        base_dir / "data" / "spam_emails.csv",
    ]
    
    dataset_path = None
    for p in possible_dataset_paths:
        if p.exists() and p.is_file():
            dataset_path = p
            break
            
    return base_dir, backend_dir, models_dir, dataset_path


def detect_columns(df: pd.DataFrame):
    """
    Intelligently detect text and label columns from diverse dataset formats
    (e.g., text/label, message/label, v2/v1, content/target, etc.).
    """
    cols = list(df.columns)
    
    # Common label column candidates
    label_candidates = ["label", "v1", "category", "target", "class", "spam", "type", "is_spam"]
    # Common text column candidates
    text_candidates = ["text", "v2", "message", "email", "content", "body", "sms", "emailtext", "email_text"]
    
    label_col = None
    text_col = None
    
    # Match label column
    for col in cols:
        if str(col).strip().lower() in label_candidates:
            label_col = col
            break
            
    # Match text column
    for col in cols:
        if str(col).strip().lower() in text_candidates:
            text_col = col
            break
            
    # Fallback heuristic if not found by name
    if not label_col:
        # Pick column with low cardinality (<= 5 unique values)
        for col in cols:
            if df[col].nunique() <= 5 and col != text_col:
                label_col = col
                break
                
    if not text_col:
        # Pick string column with highest average text length
        text_cols = [c for c in cols if c != label_col]
        if text_cols:
            text_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
            
    return text_col, label_col


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize labels into standard 'spam' and 'ham' strings."""
    def map_val(val):
        s = str(val).strip().lower()
        if s in ["1", "spam", "true", "yes", "positive", "1.0"]:
            return "spam"
        elif s in ["0", "ham", "false", "no", "not spam", "legitimate", "negative", "0.0"]:
            return "ham"
        return s
        
    return series.apply(map_val)


def load_and_clean_data(dataset_path: Path):
    """
    Loads, inspects, validates, and cleans the dataset.
    Returns cleaned X (text) and y (normalized labels) along with stats dictionary.
    """
    if not dataset_path or not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected location.\n"
            f"Please ensure 'spam_emails.csv' exists in 'backend/' or the project root."
        )
        
    print(f"==================================================")
    print(f"       EMAIL SPAM DETECTION - MODEL TRAINING      ")
    print(f"==================================================")
    print(f"[*] Loading dataset from: {dataset_path}")
    
    # Try different encodings
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(dataset_path, encoding=enc)
            print(f"[+] Loaded successfully with encoding '{enc}'")
            break
        except Exception:
            continue
            
    if df is None or df.empty:
        raise ValueError("Dataset is empty or could not be read with standard encodings.")
        
    text_col, label_col = detect_columns(df)
    if not text_col or not label_col:
        raise ValueError(
            f"Could not reliably detect text and label columns. Available columns: {list(df.columns)}"
        )
        
    print(f"[+] Detected Text Column : '{text_col}'")
    print(f"[+] Detected Label Column: '{label_col}'")
    
    raw_count = len(df)
    
    # Extract only required columns and normalize names
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "raw_label"]
    
    # Handle missing/null text values
    initial_nulls = df["text"].isna().sum() + df["raw_label"].isna().sum()
    df = df.dropna(subset=["text", "raw_label"])
    
    # Normalize labels
    df["label"] = normalize_labels(df["raw_label"])
    
    # Filter valid binary classes only ('ham' or 'spam')
    valid_classes = {"ham", "spam"}
    invalid_mask = ~df["label"].isin(valid_classes)
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        print(f"[!] Warning: Dropping {invalid_count} records with non-binary labels.")
        df = df[df["label"].isin(valid_classes)]
        
    # Clean text strings: strip whitespace
    df["text"] = df["text"].astype(str).str.strip()
    
    # Remove empty or whitespace-only texts
    empty_text_mask = df["text"] == ""
    empty_count = empty_text_mask.sum()
    if empty_count > 0:
        df = df[~empty_text_mask]
        
    # Check for duplicates
    duplicate_count = df.duplicated(subset=["text"]).sum()
    print(f"[i] Dataset contains {duplicate_count} duplicate text messages (kept for frequency distribution).")
    
    final_count = len(df)
    spam_count = (df["label"] == "spam").sum()
    ham_count = (df["label"] == "ham").sum()
    
    if spam_count == 0 or ham_count == 0:
        raise ValueError("Dataset must contain both 'spam' and 'ham' examples to train a classifier.")
        
    spam_pct = (spam_count / final_count) * 100
    ham_pct = (ham_count / final_count) * 100
    
    print("\n--------------------------------------------------")
    print("              DATASET SUMMARY & STATS             ")
    print("--------------------------------------------------")
    print(f"Total Raw Records       : {raw_count}")
    print(f"Records Removed (Nulls) : {initial_nulls + empty_count}")
    print(f"Total Valid Records     : {final_count}")
    print(f"Spam Records            : {spam_count} ({spam_pct:.2f}%)")
    print(f"Ham (Legitimate) Records: {ham_count} ({ham_pct:.2f}%)")
    print("--------------------------------------------------\n")
    
    stats = {
        "dataset_filename": dataset_path.name,
        "text_column": text_col,
        "label_column": label_col,
        "raw_records": int(raw_count),
        "valid_records": int(final_count),
        "spam_records": int(spam_count),
        "ham_records": int(ham_count),
        "spam_percentage": round(spam_pct, 2),
        "ham_percentage": round(ham_pct, 2),
        "duplicates": int(duplicate_count)
    }
    
    return df["text"], df["label"], stats


def build_pipeline():
    """
    Builds the integrated scikit-learn Pipeline with:
    1. TfidfVectorizer: unigrams + bigrams, sublinear TF scaling, English stop words.
    2. LogisticRegression: balanced class weighting, robust C parameter, reproducible seed.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        stop_words="english",
        strip_accents="unicode"
    )
    
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])
    
    return pipeline


def evaluate_and_save(pipeline, X_train, X_test, y_train, y_test, dataset_stats, models_dir: Path):
    """
    Evaluates the model on unseen test data, prints detailed metrics,
    and serializes the pipeline and metrics file.
    """
    print("[*] Training pipeline (TF-IDF + LogisticRegression)...")
    pipeline.fit(X_train, y_train)
    print("[+] Model training complete.")
    
    print("\n[*] Evaluating model on unseen test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Classes order from pipeline
    classes = list(pipeline.named_steps["classifier"].classes_)
    spam_idx = classes.index("spam")
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec_spam = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
    rec_spam = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
    f1_spam = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)
    
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Compute ROC-AUC using spam class probability
    y_test_binary = (y_test == "spam").astype(int)
    roc_auc = roc_auc_score(y_test_binary, y_proba[:, spam_idx])
    
    # Confusion Matrix: [[TN, FP], [FN, TP]] where Negative=ham, Positive=spam
    # Ensure explicit labels order ['ham', 'spam']
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    tn, fp, fn, tp = cm.ravel()
    
    clf_report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    clf_report_str = classification_report(y_test, y_pred, zero_division=0)
    
    print("\n==================================================")
    print("             MODEL EVALUATION METRICS             ")
    print("==================================================")
    print(f"Accuracy         : {acc * 100:.2f}%  (Overall classification correctness)")
    print(f"Precision (Spam) : {prec_spam * 100:.2f}%  (When predicted spam, % actually spam)")
    print(f"Recall (Spam)    : {rec_spam * 100:.2f}%  (% of all actual spam detected)")
    print(f"F1 Score (Spam)  : {f1_spam * 100:.2f}%  (Harmonic mean of precision & recall)")
    print(f"ROC-AUC Score    : {roc_auc * 100:.2f}%  (Class separability score)")
    print("--------------------------------------------------")
    print("Confusion Matrix:")
    print(f"  True Negatives (Legitimate Ham identified as Ham): {tn}")
    print(f"  False Positives (Legitimate Ham misclassified as Spam): {fp}")
    print(f"  False Negatives (Spam missed as Ham): {fn}")
    print(f"  True Positives (Spam correctly caught): {tp}")
    print("--------------------------------------------------")
    print("Classification Report:\n")
    print(clf_report_str)
    print("==================================================\n")
    
    # Save model pipeline
    model_path = models_dir / "spam_classifier.pkl"
    joblib.dump(pipeline, model_path)
    print(f"[+] Saved trained pipeline to: {model_path}")
    
    # Package metrics dictionary
    metrics_data = {
        "model_name": "Email Spam Classifier",
        "algorithm": "Logistic Regression with L2 Regularization",
        "vectorizer": "TF-IDF (Term Frequency-Inverse Document Frequency)",
        "features": {
            "ngram_range": [1, 2],
            "max_features": 5000,
            "sublinear_tf": True,
            "stop_words": "english"
        },
        "dataset": dataset_stats,
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "performance": {
            "accuracy": round(float(acc) * 100, 2),
            "precision_spam": round(float(prec_spam) * 100, 2),
            "recall_spam": round(float(rec_spam) * 100, 2),
            "f1_spam": round(float(f1_spam) * 100, 2),
            "precision_macro": round(float(prec_macro) * 100, 2),
            "recall_macro": round(float(rec_macro) * 100, 2),
            "f1_macro": round(float(f1_macro) * 100, 2),
            "roc_auc": round(float(roc_auc) * 100, 2)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": clf_report_dict
    }
    
    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[+] Saved model performance metrics to: {metrics_path}")
    
    return metrics_data


def main():
    try:
        base_dir, backend_dir, models_dir, dataset_path = get_project_paths()
        
        # 1. Load & Clean Data
        X, y, dataset_stats = load_and_clean_data(dataset_path)
        
        # 2. Stratified Train/Test Split
        print(f"[*] Splitting dataset (80% train, 20% test, stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"[+] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
        
        # 3. Build & Train Pipeline
        pipeline = build_pipeline()
        
        # 4. Evaluate & Save Artifacts
        evaluate_and_save(pipeline, X_train, X_test, y_train, y_test, dataset_stats, models_dir)
        
        print("\n[SUCCESS] Training script completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n[!] Training Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
