"""
preprocess.py - Shared preprocessing for IBM Telco Customer Churn dataset.

Handles all 21 dataset columns using manual Pandas operations.
No sklearn Pipeline or ColumnTransformer is used.

Columns handled:
  - Drop: customerID
  - Target: Churn (Yes->1, No->0)
  - Binary encode: gender, Partner, Dependents, PhoneService, PaperlessBilling
  - Service columns (map "No internet service"/"No phone service" -> "No", then binary):
      MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
      TechSupport, StreamingTV, StreamingMovies
  - Numeric: SeniorCitizen (already 0/1), tenure, MonthlyCharges, TotalCharges
  - One-hot encode (drop_first=True): InternetService, Contract, PaymentMethod
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# -- Column definitions ------------------------------------------------------

# Binary columns: map Yes/No or Male/Female to 1/0
BINARY_COLUMNS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

BINARY_MAP = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
}

# Service columns that may have "No internet service" or "No phone service"
# These get mapped to "No" first, then binary encoded (Yes=1, No=0)
SERVICE_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Categorical columns for one-hot encoding (pd.get_dummies with drop_first=True)
CATEGORICAL_COLUMNS = ["InternetService", "Contract", "PaymentMethod"]

# Numeric columns (no transformation needed, already numeric)
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


# -- Data loading & validation -----------------------------------------------

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def load_and_validate(path: str) -> pd.DataFrame:
    """
    Step 1 - Data Validation: Load CSV and verify all expected columns exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    print(f"[Validation] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns [OK]")
    return df


# -- Cleaning ----------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2 - Data Cleaning:
      - Drop customerID
      - Convert TotalCharges to numeric (some rows have whitespace strings)
      - Fill missing TotalCharges with median
      - Convert Churn: No->0, Yes->1
    """
    df = df.copy()

    # Drop customerID - not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("[Cleaning] Dropped customerID [OK]")

    # TotalCharges: some rows have " " (whitespace), convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isna().sum()
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    print(f"[Cleaning] TotalCharges: {missing_tc} missing values filled with median ({median_tc:.2f}) [OK]")

    # Encode target: Churn
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    print(f"[Cleaning] Churn encoded: {df['Churn'].value_counts().to_dict()} [OK]")

    return df


# -- Feature engineering -----------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4 - Feature Engineering:
      - Binary-encode gender, Partner, Dependents, PhoneService, PaperlessBilling
      - Normalize service columns (replace "No internet/phone service" with "No")
      - Binary-encode service columns
      - One-hot encode InternetService, Contract, PaymentMethod (drop_first=True)
    """
    df = df.copy()

    # Binary encoding for simple Yes/No and Male/Female columns
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP).fillna(0).astype(int)
    print(f"[Features] Binary encoded: {BINARY_COLUMNS} [OK]")

    # Service columns: normalize "No internet service" / "No phone service" -> "No"
    for col in SERVICE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace({
                "No internet service": "No",
                "No phone service": "No",
            })
            # Now binary encode: Yes=1, No=0
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    print(f"[Features] Service columns normalized & binary encoded: {SERVICE_COLUMNS} [OK]")

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    print(f"[Features] One-hot encoded: {CATEGORICAL_COLUMNS} (drop_first=True) [OK]")

    # Ensure all numeric columns are float
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


# -- Full training preprocessing pipeline ------------------------------------

def preprocess_training_data(csv_path: str):
    """
    Run the full preprocessing pipeline for training.
    Returns X, y, and the list of feature column names (for saving).
    """
    # Step 1: Load and validate
    df = load_and_validate(csv_path)

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: EDA summary (printed)
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution (Churn):\n{df['Churn'].value_counts()}")
    print(f"\nChurn rate: {df['Churn'].mean() * 100:.2f}%")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    if df.isnull().sum().sum() == 0:
        print("  No missing values [OK]")
    print(f"\nNumeric column statistics:")
    print(df[NUMERIC_COLUMNS].describe().round(2))
    print("=" * 60 + "\n")

    # Step 4: Feature engineering
    df = encode_features(df)

    # Separate features and target
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Store feature column names for alignment during prediction
    feature_columns = X.columns.tolist()

    print(f"[Pipeline] Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, feature_columns


# -- Single-row preprocessing for FastAPI prediction -------------------------

def preprocess_single_row(data: dict, feature_columns: list) -> pd.DataFrame:
    """
    Preprocess a single customer record for prediction.
    Applies the exact same transformations as training, then aligns
    columns with the saved training feature list to avoid mismatch.

    Args:
        data: dict with original dataset columns (except customerID and Churn)
        feature_columns: list of feature names from training

    Returns:
        pd.DataFrame with one row, columns aligned to training features
    """
    df = pd.DataFrame([data])

    # Convert TotalCharges to numeric, fill missing with 0 (single row - no median available)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Binary encode
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP).fillna(0).astype(int)

    # Service columns: normalize and binary encode
    for col in SERVICE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace({
                "No internet service": "No",
                "No phone service": "No",
            })
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # Align columns with training features - add missing columns as 0, drop extra ones
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df.astype(float)


# -- Feature columns save/load -----------------------------------------------

def save_feature_columns(feature_columns: list, model_dir: str):
    """Save the training feature column names to JSON for prediction alignment."""
    path = Path(model_dir) / "feature_columns.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"[Save] Feature columns saved to {path} ({len(feature_columns)} features) [OK]")


def load_feature_columns(model_dir: str) -> list:
    """Load the training feature column names from JSON."""
    path = Path(model_dir) / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(
            f"feature_columns.json not found in {model_dir}. Run app.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



