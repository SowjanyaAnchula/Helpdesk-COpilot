"""
train_setfit.py — HelpDesk Copilot
Model 2: MiniLM sentence embeddings + SVM classifier
Model 3: SetFit few-shot result (logged from prior run)
Comparison against TF-IDF + LR baseline.
"""

import os
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── PATHS ──
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── CONFIG ──
MODEL_NAME  = "BAAI/bge-small-en-v1.5"
RANDOM_SEED = 42
TEST_SIZE   = 0.2


def load_data(path):
    df = pd.read_csv(path)
    df = df[df["language"] == "en"].dropna(subset=["text", "queue", "priority"])
    print(f"Loaded {len(df):,} English tickets")
    return df


if __name__ == "__main__":
    # ── Load data ──
    df = load_data(os.path.join(DATA_DIR, "tickets_clean.csv"))

    # Encode labels
    queue_enc    = LabelEncoder()
    priority_enc = LabelEncoder()
    df["queue_id"]    = queue_enc.fit_transform(df["queue"])
    df["priority_id"] = priority_enc.fit_transform(df["priority"])

    queue_labels    = queue_enc.classes_.tolist()
    priority_labels = priority_enc.classes_.tolist()
    print(f"Queue classes ({len(queue_labels)}):    {queue_labels}")
    print(f"Priority classes ({len(priority_labels)}): {priority_labels}")

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df["queue_id"]
    )
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── Generate embeddings ──
    print(f"\nLoading sentence transformer: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    print("Encoding training set...")
    X_train = embedder.encode(
        train_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # BGE works best with normalized embeddings
    )

    print("Encoding test set...")
    X_test = embedder.encode(
        test_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Embedding shape: {X_train.shape}")

    # ══════════════════════════════════════════
    #  QUEUE CLASSIFIER — SVM with RBF kernel
    # ══════════════════════════════════════════
    print("\n" + "=" * 50)
    print("Training Queue Classifier (SVM)")
    print("=" * 50)

    y_train_q = train_df["queue_id"].values
    y_test_q  = test_df["queue_id"].values

    queue_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        C=10.0,
        gamma="scale",
        random_state=RANDOM_SEED,
        probability=True,       # needed for calibration & confidence scores
    )

    print("Fitting SVM (this takes 2-3 min)...")
    queue_svm.fit(X_train, y_train_q)

    # Evaluate queue
    y_pred_q = queue_svm.predict(X_test)
    acc_q    = accuracy_score(y_test_q, y_pred_q)
    report_q = classification_report(
        y_test_q, y_pred_q, target_names=queue_labels, zero_division=0
    )
    macro_f1_q = float(report_q.split("macro avg")[1].split()[2])

    print(f"Accuracy:  {acc_q:.4f}")
    print(f"Macro-F1:  {macro_f1_q:.4f}")
    print(f"\n{report_q}")

    # ══════════════════════════════════════════
    #  PRIORITY CLASSIFIER — SVM with RBF kernel
    # ══════════════════════════════════════════
    print("\n" + "=" * 50)
    print("Training Priority Classifier (SVM)")
    print("=" * 50)

    y_train_p = train_df["priority_id"].values
    y_test_p  = test_df["priority_id"].values

    priority_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        C=10.0,
        gamma="scale",
        random_state=RANDOM_SEED,
        probability=True,
    )

    print("Fitting SVM...")
    priority_svm.fit(X_train, y_train_p)

    y_pred_p = priority_svm.predict(X_test)
    acc_p    = accuracy_score(y_test_p, y_pred_p)
    report_p = classification_report(
        y_test_p, y_pred_p, target_names=priority_labels, zero_division=0
    )
    macro_f1_p = float(report_p.split("macro avg")[1].split()[2])

    print(f"Accuracy:  {acc_p:.4f}")
    print(f"Macro-F1:  {macro_f1_p:.4f}")
    print(f"\n{report_p}")

    # ══════════════════════════════════════════
    #  FINAL COMPARISON TABLE
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<35} {'Queue F1':>10} {'Priority F1':>12}")
    print("-" * 59)
    print(f"{'TF-IDF + LR (Baseline)':<35} {'0.6000':>10} {'0.6300':>12}")
    print(f"{'SetFit few-shot (MiniLM)':<35} {'0.2500':>10} {'   N/A':>12}")
    print(f"{'BGE + SVM (This run)':<35} {macro_f1_q:>10.4f} {macro_f1_p:>12.4f}")
    print("=" * 60)

    if macro_f1_q > 0.60:
        print(f"\n🎉 BGE + SVM BEATS BASELINE by {(macro_f1_q - 0.60)*100:+.1f} pp on Queue!")
    else:
        print(f"\n📊 TF-IDF baseline still leads on Queue by {(0.60 - macro_f1_q)*100:.1f} pp")
        print("   This is expected — support tickets are keyword-driven.")

    # ══════════════════════════════════════════
    #  SAVE MODELS
    # ══════════════════════════════════════════
    save_path = os.path.join(MODEL_DIR, "bge_svm")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "queue_classifier.pkl"), "wb") as f:
        pickle.dump({
            "model":   queue_svm,
            "encoder": queue_enc,
            "labels":  queue_labels,
        }, f)

    with open(os.path.join(save_path, "priority_classifier.pkl"), "wb") as f:
        pickle.dump({
            "model":   priority_svm,
            "encoder": priority_enc,
            "labels":  priority_labels,
        }, f)

    with open(os.path.join(save_path, "config.pkl"), "wb") as f:
        pickle.dump({"embedder": MODEL_NAME}, f)

    print(f"\n✅ Queue classifier saved to {save_path}/queue_classifier.pkl")
    print(f"✅ Priority classifier saved to {save_path}/priority_classifier.pkl")
    print(f"✅ Ready for agent integration.")