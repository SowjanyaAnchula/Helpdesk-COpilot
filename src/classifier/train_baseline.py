"""
train_baseline.py — HelpDesk Copilot
Baseline classifier: TF-IDF + Logistic Regression
Predicts queue (department) and priority from ticket text.
"""

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

# ── PATHS ──
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── CONFIG ──
TEST_SIZE   = 0.2
RANDOM_SEED = 42
MAX_ITER    = 1000


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} tickets")

    # Keep only English tickets for baseline
    df = df[df["language"] == "en"]
    print(f"English only: {len(df):,} tickets")

    # Drop rows with missing text, queue or priority
    df = df.dropna(subset=["text", "queue", "priority"])
    print(f"After dropping nulls: {len(df):,} tickets")

    return df


def build_pipeline(class_weight="balanced") -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),   # unigrams + bigrams
            sublinear_tf=True,    # apply log normalization
            min_df=2,             # ignore very rare terms
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=MAX_ITER,
            random_state=RANDOM_SEED,
            C=5.0,
        )),
    ])


def evaluate(name: str, y_true, y_pred, labels) -> dict:
    acc     = accuracy_score(y_true, y_pred)
    report  = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    macro_f1 = float(report.split("macro avg")[1].split()[2])

    print(f"\n{'='*50}")
    print(f"{name} Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    return {"accuracy": acc, "macro_f1": macro_f1}


def train_and_save(df: pd.DataFrame, target: str, model_name: str) -> dict:
    print(f"\n{'#'*50}")
    print(f"Training: {model_name}  |  Target: {target}")
    print(f"{'#'*50}")

    X = df["text"]
    y = df[target]
    labels = sorted(y.unique().tolist())

    print(f"Classes ({len(labels)}): {labels}")
    print(f"Distribution:\n{y.value_counts()}\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Build and train pipeline
    pipeline = build_pipeline(class_weight="balanced")
    pipeline.fit(X_train, y_train)

    # Calibrate probabilities with Platt scaling
    # (needed for reliable confidence scores → escalation logic)
    calibrated = CalibratedClassifierCV(pipeline, cv=5, method="sigmoid")
    calibrated.fit(X_train, y_train)

    # Evaluate
    y_pred  = calibrated.predict(X_test)
    metrics = evaluate(model_name, y_test, y_pred, labels)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": calibrated, "labels": labels}, f)
    print(f"\n✅ Saved to {model_path}")

    return metrics


if __name__ == "__main__":
    # Load
    tickets_path = os.path.join(DATA_DIR, "tickets_clean.csv")
    df = load_data(tickets_path)

    # Train queue classifier
    queue_metrics = train_and_save(df, target="queue", model_name="queue_classifier_baseline")

    # Train priority classifier
    priority_metrics = train_and_save(df, target="priority", model_name="priority_classifier_baseline")

    # Summary
    print("\n" + "="*50)
    print("BASELINE RESULTS SUMMARY")
    print("="*50)
    print(f"Queue    — Accuracy: {queue_metrics['accuracy']:.4f}  |  Macro-F1: {queue_metrics['macro_f1']:.4f}")
    print(f"Priority — Accuracy: {priority_metrics['accuracy']:.4f}  |  Macro-F1: {priority_metrics['macro_f1']:.4f}")
    print("\nNext step: train SetFit model and compare.")