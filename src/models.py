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
from sklearn.tree import DecisionTreeClassifier
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
    output_dir: str = "Reports/Results/Logistic Regression",
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



def train_decision_tree(
    processed_csv_path: str = "data/processed/loan_data_processed.csv",
    output_dir: str = "Reports/Results/decision_tree",
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = None,
):
    """
    Train a simple Decision Tree classifier.

    Saves artifacts to output_dir:
    - decision_tree_model.joblib (model, feature_columns)
    - metrics.json
    - classification_report.txt
    - confusion_matrix.png
    - test_predictions.csv
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(processed_csv_path)

    if "Loan_Status" not in df.columns:
        raise KeyError("Loan_Status column not found in processed dataset.")

    # 2) Target
    y = _binary_target(df["Loan_Status"])

    # 3) Features (no scaling required for tree models)
    X = _prepare_features(df)

    # 4) Train/test split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5) Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    # 6) Predict & evaluate
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }

    # 7) Save model and artifacts
    artifact = {
        "model": clf,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(artifact, outdir / "decision_tree_model.joblib")

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(outdir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    }, index=y_test.index)
    preds_df.to_csv(outdir / "test_predictions.csv", index=True)

    # 😎 Save confusion matrix image
    _save_confusion_matrix_png(cm, str(outdir / "confusion_matrix.png"))

    print("Decision Tree training finished.")
    print("Saved model and artifacts to:", outdir)
    print("Metrics:", metrics)

    return {
        "model": clf,
        "metrics": metrics,
        "feature_columns": X.columns.tolist(),
    }

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop target column and ID column if present.
    One-hot encode categorical columns using pandas.get_dummies (drop_first=True).
    Returns processed feature DataFrame.
    """
    X = df.copy()
    if "Loan_Status" in X.columns:
        X = X.drop(columns=["Loan_Status"])
    if "Loan_ID" in X.columns:
        X = X.drop(columns=["Loan_ID"])

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X