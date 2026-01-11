# linear regression moded

from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib


def _binary_target(series: pd.Series) -> pd.Series:
    """
    Simple target encoding:
    Treat values that start with 'Y' or 'y' as 1, everything else as 0.
    """
    return series.astype(str).str.strip().str.upper().str.startswith("Y").astype(int)


def _save_confusion_matrix_png(cm: np.ndarray, output_path: str) -> None:
    """
    Save a simple confusion matrix heatmap to output_path.
    cm is the 2x2 confusion matrix array.
    """
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_logistic_regression(
    processed_csv_path: str = "data/processed/loan_data_processed.csv",
    output_dir: str = "artifacts/models",
    test_size: float = 0.2,
    random_state: int = 42,
):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(processed_csv_path)

    if "Loan_Status" not in df.columns:
        raise KeyError("Loan_Status column not found in processed dataset.")

    # 2) Prepare target
    y = _binary_target(df["Loan_Status"])

    # 3) Prepare features: drop target and any ID column if present
    X = df.drop(columns=["Loan_Status"])
    if "Loan_ID" in X.columns:
        X = X.drop(columns=["Loan_ID"])

    # 4) One-hot encode categorical columns (simple)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 5) Scale numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # 6) Train / test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 7) Train logistic regression
    model = LogisticRegression(solver="liblinear", random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)

    #  Evaluate
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        y_proba = None
        roc = None

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc,
        "confusion_matrix": cm.tolist(),  # save as list for JSON
    }

    # 9) Save model artifacts
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(artifacts, outdir / "logistic_regression_simple.joblib")

    # 10) Save metrics and classification report
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(outdir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # 11) Save test predictions
    preds = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    }, index=y_test.index)
    preds.to_csv(outdir / "test_predictions.csv", index=True)

    # 12) Save confusion matrix image
    _save_confusion_matrix_png(cm, str(outdir / "confusion_matrix.png"))

    print("Training finished.")
    print("Saved model and artifacts to:", outdir)
    print("Metrics:", metrics)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_columns": X.columns.tolist(),
    }

# -----------------------------end linear regression----------------------------
