
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
from sklearn.ensemble import RandomForestClassifier
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

# Logistic Regression Model

# Function to convert the target variable into binary format
# Approved loans are mapped to 1, rejected loans to 0
def _binary_target(series: pd.Series) -> pd.Series:
    """
    Simple target encoding:
    Treat values that start with 'Y' or 'y' as 1, everything else as 0.
    """
    return series.astype(str).str.strip().str.upper().str.startswith("Y").astype(int)


# Function to save a confusion matrix as an image
# This helps visually evaluate model performance
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


# Main function to train and evaluate a Logistic Regression Model
def train_logistic_regression(
    processed_csv_path: str = "data/processed/loan_data_processed.csv",
    output_dir: str = "Reports/Results/Logistic Regression",
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Create output directory if it doesn't exist
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load the processed data
    df = pd.read_csv(processed_csv_path)

    if "Loan_Status" not in df.columns:
        raise KeyError("Loan_Status column not found in processed dataset.")

    # 2) Prepare target variable
    # Convert loan approval status to binary values
    y = _binary_target(df["Loan_Status"])

    # 3) Prepare features: drop target and any ID column if present
    X = df.drop(columns=["Loan_Status"])
    if "Loan_ID" in X.columns:
        X = X.drop(columns=["Loan_ID"])

    # 4) One-hot encode categorical columns
    # This converts text categories into numeric format
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 5) Scale numerical features
    # Logistic Regression performs better when features are on similar scales
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # 6) Train / test split
    # Stratify ensures class balance is preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 7) Train the Logistic Regression Model
    model = LogisticRegression(solver="liblinear", random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)

    #  Evaluate model performance on test data
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        y_proba = None
        roc = None

    # Creating the confusion matrix
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


# Decision Tree Model

# Function to train and evaluate a Decision Tree Classifier
def train_decision_tree(
    processed_csv_path: str = "data/processed/loan_data_processed.csv",
    output_dir: str = "Reports/Results/Decision Tree",
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

    # 3) Prepare feature matrix
    X = _prepare_features(df)

    # 4) Train/test split (stratify)
    # Stratify keeps the approval/rejection ratio similar in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5) Train Decision Tree
    # max_depth is used to prevent overfitting
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

    # 8) Save confusion matrix image
    _save_confusion_matrix_png(cm, str(outdir / "confusion_matrix.png"))

    print("Decision Tree training finished.")
    print("Saved model and artifacts to:", outdir)
    print("Metrics:", metrics)

    return {
        "model": clf,
        "metrics": metrics,
        "feature_columns": X.columns.tolist(),
    }


# Function to prepare feature matrix for ML models
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





# Random Forest Model


# Function to train and evaluate a Random Forest classifier
# This model is an extension of Decision Trees and usually gives better performance
def train_random_forest(
    processed_csv_path: str = "data/processed/loan_data_processed.csv",
    output_dir: str = "Reports/Results/Random Forest",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = None,
):
    """
    Train a simple Random Forest classifier.

    Keeps everything beginner-friendly and consistent with the other trainers:
    - One-hot encode categoricals using pandas.get_dummies
    - No special scaling (tree-based model)
    - Compute Accuracy, Precision, Recall, F1, Confusion Matrix
    - Save model and artifacts:
      - random_forest_model.joblib
      - metrics.json
      - classification_report.txt
      - confusion_matrix.png
      - test_predictions.csv
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load processed data
    df = pd.read_csv(processed_csv_path)

    if "Loan_Status" not in df.columns:
        raise KeyError("Loan_Status column not found in processed dataset.")

    # 2) Prepare target variable
    # Convert loan approval status into binary values
    y = _binary_target(df["Loan_Status"])

    # 3) Preparing feature matrix
    X = _prepare_features(df)

    # 4) Train/test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5) Train Random Forest
    # Building multiple decision trees and averages their results
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 6) Predict & evaluate
    y_pred = rf.predict(X_test)


    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }

    # 7) Save model and artifacts
    artifact = {
        "model": rf,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(artifact, outdir / "random_forest_model.joblib")

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(outdir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    }, index=y_test.index)
    preds_df.to_csv(outdir / "test_predictions.csv", index=True)

    # 8) Save confusion matrix image (counts)
    _save_confusion_matrix_png(cm, str(outdir / "confusion_matrix.png"))

    print("Random Forest training finished.")
    print("Saved model and artifacts to:", outdir)
    print("Metrics:", metrics)

    return {
        "model": rf,
        "metrics": metrics,
        "feature_columns": X.columns.tolist(),
    }