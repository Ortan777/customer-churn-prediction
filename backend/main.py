"""
main.py — FastAPI Application for Customer Churn Prediction

Endpoints:
    GET  /         — Welcome message
    GET  /health   — Health check
    POST /predict  — Predict churn for a single customer

The POST /predict endpoint accepts all original dataset columns
except customerID and Churn, preprocesses them exactly like training,
aligns columns with the saved training feature list, and returns:

    {
        "prediction": "Yes" or "No",
        "probability": float (0.0 to 1.0)
    }
"""

import os
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from preprocess import preprocess_single_row, load_feature_columns

# -- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "customer_churn_model.joblib")

# -- Global model & feature columns (loaded once at startup) ----------------
MODEL = None
FEATURE_COLUMNS = None


# -- Pydantic schemas --------------------------------------------------------

class CustomerInput(BaseModel):
    """
    Input schema for POST /predict.
    Accepts all original dataset columns EXCEPT customerID and Churn.
    """
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., description="Number of months with company")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    MonthlyCharges: float = Field(..., description="Monthly charges in dollars")
    TotalCharges: float = Field(..., description="Total charges in dollars")


class PredictionResponse(BaseModel):
    """
    Response schema for POST /predict.
    """
    prediction: str = Field(..., description="Yes or No")
    probability: float = Field(..., description="Churn probability (0.0 to 1.0)")


# -- FastAPI app -------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict customer churn using an XGBoost model trained on the "
        "IBM Telco Customer Churn dataset. Built with FastAPI, scikit-learn, "
        "and XGBoost."
    ),
    version="1.0.0",
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Startup event: load model ----------------------------------------------

@app.on_event("startup")
def load_model_on_startup():
    """
    Load the trained model and feature columns at startup.
    This avoids loading the model on every request.
    """
    global MODEL, FEATURE_COLUMNS

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at {MODEL_PATH}. Run app.py first to train.")
        return

    MODEL = joblib.load(MODEL_PATH)
    FEATURE_COLUMNS = load_feature_columns(MODEL_DIR)
    print(f"[Startup] Model loaded from {MODEL_PATH} [OK]")
    print(f"[Startup] Feature columns loaded ({len(FEATURE_COLUMNS)} features) [OK]")


# -- Endpoints ---------------------------------------------------------------

@app.get("/")
def root():
    """
    GET / — Welcome message.
    """
    return {
        "message": "Customer Churn Prediction API",
        "description": "Use POST /predict to get churn predictions",
        "endpoints": {
            "GET /": "This welcome message",
            "GET /health": "Health check",
            "POST /predict": "Predict customer churn",
        },
    }


@app.get("/health")
def health_check():
    """
    GET /health — Health check endpoint.
    Returns the status of the API and whether the model is loaded.
    """
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "features_loaded": FEATURE_COLUMNS is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """
    POST /predict — Predict churn for a single customer.

    Accepts all original dataset columns except customerID and Churn.
    Preprocesses them exactly like training, aligns columns with the
    saved training feature list, and returns prediction + probability.
    """
    # Check if model is loaded
    if MODEL is None or FEATURE_COLUMNS is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first by running: python app.py",
        )

    try:
        # Convert Pydantic model to dict
        customer_data = customer.dict()

        # Preprocess exactly like training — uses the same preprocess module
        features = preprocess_single_row(customer_data, FEATURE_COLUMNS)

        # Get prediction and probability
        prediction = MODEL.predict(features)[0]
        probability = float(MODEL.predict_proba(features)[0][1])

        # Map prediction back to Yes/No
        prediction_label = "Yes" if prediction == 1 else "No"

        return PredictionResponse(
            prediction=prediction_label,
            probability=round(probability, 4),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# -- Run directly for development -------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


