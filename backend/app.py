"""
app.py - Customer Churn Prediction Training Pipeline

Complete training pipeline for the IBM Telco Customer Churn dataset.
Uses manual Pandas preprocessing (no sklearn Pipeline or ColumnTransformer).

Pipeline stages:
  1. Data Validation
  2. Data Cleaning
  3. EDA (Exploratory Data Analysis)
  4. Feature Engineering
  5. Train/Test Split (80/20)
  6. Baseline Model (Logistic Regression)
  7. XGBoost Model
  8. Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC)
  9. MLflow Tracking
  10. Save Best Model (joblib)

Usage:
    python app.py
"""

import os
import sys
import warnings
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from xgboost import XGBClassifier

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import our custom preprocessing module
from preprocess import (
    preprocess_training_data,
    save_feature_columns,
)

# -- MLflow setup (optional - works even if MLflow server is not running) ----
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False
    print("[Warning] MLflow not installed. Experiment tracking disabled.")

# -- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco Customer Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "customer_churn_model.joblib")
MLFLOW_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db").replace("\\", "/")


# -- Evaluation --------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and print all metrics.

    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Print evaluation results
    print(f"\n{'=' * 60}")
    print(f"  {model_name} - EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"{'=' * 60}\n")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist(),
    }
    return metrics


# -- MLflow logging ----------------------------------------------------------

def log_to_mlflow(model, model_name, params, metrics, is_xgboost=False):
    """
    Log a training run to MLflow with parameters, metrics, and model artifact.
    """
    if not MLFLOW_AVAILABLE:
        print(f"[MLflow] Skipped (MLflow not available)")
        return

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics (exclude confusion matrix - it's a list, not a scalar)
        scalar_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        mlflow.log_metrics(scalar_metrics)

        # Log confusion matrix as artifact
        cm_data = json.dumps({"confusion_matrix": metrics["confusion_matrix"]})
        mlflow.log_text(cm_data, "confusion_matrix.json")

        # Log the model
        try:
            # MLflow v3: use 'name' parameter (artifact_path is deprecated)
            if is_xgboost:
                mlflow.xgboost.log_model(model, name="xgboost_model")
            else:
                mlflow.sklearn.log_model(model, name="sklearn_model")
        except TypeError:
            # Fallback for MLflow v2: use 'artifact_path' parameter
            if is_xgboost:
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[MLflow] Run logged for {model_name} [OK]")


# -- S3 upload (optional) ---------------------------------------------------

def upload_to_s3(local_path, bucket_name, s3_key):
    """
    Upload model artifacts to AWS S3.
    Set environment variables: AWS_S3_BUCKET, AWS_REGION
    """
    try:
        import boto3
        s3_client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"[S3] Uploaded {local_path} -> s3://{bucket_name}/{s3_key} [OK]")
    except ImportError:
        print("[S3] boto3 not installed. Skipping S3 upload.")
    except Exception as e:
        print(f"[S3] Upload failed: {e}")


# -- Main training pipeline -------------------------------------------------

def main():
    """
    Run the complete training pipeline.
    """
    print("\n" + "#" * 60)
    print("#  CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("#" * 60 + "\n")

    # -- Step 1-4: Preprocessing (validation, cleaning, EDA, feature engineering)
    print("=" * 60)
    print("STEP 1-4: DATA PREPROCESSING")
    print("=" * 60)
    X, y, feature_columns = preprocess_training_data(DATA_PATH)

    # -- Step 5: Train/Test Split (80/20) ------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: TRAIN/TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"  Training set : {X_train.shape[0]} samples")
    print(f"  Test set     : {X_test.shape[0]} samples")
    print(f"  Features     : {X_train.shape[1]}")
    print(f"  Train churn rate: {y_train.mean() * 100:.2f}%")
    print(f"  Test churn rate : {y_test.mean() * 100:.2f}%")

    # -- Setup MLflow --------------------------------------------------------
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("telco-customer-churn")
        print(f"\n[MLflow] Tracking URI: {MLFLOW_URI}")
        print(f"[MLflow] Experiment: telco-customer-churn")

    # -- Step 6: Baseline Model (Logistic Regression) ------------------------
    print("\n" + "=" * 60)
    print("STEP 6: BASELINE MODEL - LOGISTIC REGRESSION")
    print("=" * 60)

    lr_params = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "random_state": 42,
        "solver": "lbfgs",
    }

    lr_model = LogisticRegression(
        max_iter=lr_params["max_iter"],
        random_state=lr_params["random_state"],
        solver=lr_params["solver"],
    )
    lr_model.fit(X_train, y_train)
    print("[Training] Logistic Regression trained [OK]")

    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression (Baseline)")

    # Log baseline to MLflow
    log_to_mlflow(lr_model, "LogisticRegression_Baseline", lr_params, lr_metrics)

    # -- Step 7: XGBoost Model -----------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: XGBOOST MODEL")
    print("=" * 60)

    xgb_params = {
        "model_type": "XGBClassifier",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": "False",
    }

    xgb_model = XGBClassifier(
        n_estimators=int(xgb_params["n_estimators"]),
        max_depth=int(xgb_params["max_depth"]),
        learning_rate=float(xgb_params["learning_rate"]),
        subsample=float(xgb_params["subsample"]),
        colsample_bytree=float(xgb_params["colsample_bytree"]),
        random_state=int(xgb_params["random_state"]),
        eval_metric="logloss",
        use_label_encoder=False,
    )
    xgb_model.fit(X_train, y_train)
    print("[Training] XGBoost trained [OK]")

    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Log XGBoost to MLflow
    log_to_mlflow(xgb_model, "XGBoost_Final", xgb_params, xgb_metrics, is_xgboost=True)

    # -- Step 8: Model Comparison --------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Logistic Regression':>20} {'XGBoost':>15}")
    print("-" * 50)
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        lr_val = lr_metrics[metric]
        xgb_val = xgb_metrics[metric]
        winner = " <-" if xgb_val >= lr_val else ""
        print(f"{metric:<15} {lr_val:>20.4f} {xgb_val:>15.4f}{winner}")
    print("-" * 50)

    # -- Step 9: Save Best Model (XGBoost) -----------------------------------
    print("\n" + "=" * 60)
    print("STEP 9: SAVE BEST MODEL")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the XGBoost model using joblib
    joblib.dump(xgb_model, MODEL_PATH)
    print(f"[Save] XGBoost model saved to {MODEL_PATH} [OK]")

    # Save feature column names for prediction alignment
    save_feature_columns(feature_columns, MODEL_DIR)

    # -- Step 10: Upload to S3 (optional) ------------------------------------
    s3_bucket = os.environ.get("AWS_S3_BUCKET")
    if s3_bucket:
        print("\n" + "=" * 60)
        print("STEP 10: UPLOAD TO AWS S3")
        print("=" * 60)
        upload_to_s3(MODEL_PATH, s3_bucket, "models/customer_churn_model.joblib")
        fc_path = os.path.join(MODEL_DIR, "feature_columns.json")
        upload_to_s3(fc_path, s3_bucket, "models/feature_columns.json")
    else:
        print("\n[S3] AWS_S3_BUCKET not set. Skipping S3 upload.")
        print("     Set AWS_S3_BUCKET environment variable to enable.")

    # -- Summary -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("#  TRAINING COMPLETE!")
    print("#" * 60)
    print(f"  Model     : XGBoost (best)")
    print(f"  Accuracy  : {xgb_metrics['accuracy']:.4f}")
    print(f"  F1 Score  : {xgb_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {xgb_metrics['roc_auc']:.4f}")
    print(f"  Saved to  : {MODEL_PATH}")
    print(f"  Features  : {len(feature_columns)} columns")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()





