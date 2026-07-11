# Customer Churn Prediction — MLOps Project

End-to-end Customer Churn Prediction pipeline using the **IBM Telco Customer Churn** dataset.

## Pipeline

```
CSV Dataset → Data Validation → Data Cleaning → EDA → Feature Engineering
→ Train/Test Split → Baseline (Logistic Regression) → XGBoost Model
→ Evaluation → MLflow Tracking → Save Best Model → FastAPI → Docker
→ AWS Elastic Beanstalk
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| ML Models | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Model Serialization | joblib |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker, Docker Compose |
| Cloud Deployment | AWS Elastic Beanstalk |
| Artifact Storage | AWS S3 |

## Project Structure

```
customer_churn/
├── backend/
│   ├── data/
│   │   └── Telco Customer Churn.csv
│   ├── models/                          # Generated after training
│   │   ├── customer_churn_model.joblib
│   │   └── feature_columns.json
│   ├── .ebextensions/
│   │   └── 01_packages.config
│   ├── .platform/
│   │   └── nginx/conf.d/proxy.conf
│   ├── app.py                           # Training pipeline
│   ├── preprocess.py                    # Shared preprocessing
│   ├── main.py                          # FastAPI application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── Dockerrun.aws.json
├── frontend/                            # React dashboard
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python app.py
```

This will:
- Load and validate the Telco Customer Churn CSV (21 columns)
- Clean data (drop customerID, convert TotalCharges, encode Churn)
- Perform EDA (print dataset stats, class distribution, missing values)
- Engineer features (binary encode, one-hot encode with pd.get_dummies)
- Split 80/20 (stratified)
- Train Logistic Regression baseline
- Train XGBoost model
- Evaluate both (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC)
- Log experiments to MLflow
- Save XGBoost model to `models/customer_churn_model.joblib`
- Save feature columns to `models/feature_columns.json`

### 3. Run the API

```bash
uvicorn main:app --reload --port 8000
```

### 4. Test Endpoints

**GET /** — Welcome message:
```bash
curl http://localhost:8000/
```

**GET /health** — Health check:
```bash
curl http://localhost:8000/health
```

**POST /predict** — Predict churn:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
  }'
```

**Response:**
```json
{
  "prediction": "Yes",
  "probability": 0.8234
}
```

### 5. MLflow UI

```bash
cd backend
mlflow ui --backend-store-uri mlruns
```
Open http://localhost:5000 to view experiment runs.

### 6. Run the Frontend

If you have Node.js and `npm` installed, you can start the React development server:
```bash
cd frontend
npm install
npm run dev
```

Alternatively, you can run a lightweight static Python server to view the standalone frontend page:
```bash
cd frontend
python -m http.server 5173
```
Now open **http://localhost:5173** in your web browser.

## Docker

### Build and run locally:
```bash
docker-compose up --build
```

### Build backend only:
```bash
cd backend
docker build -t customer-churn-api .
docker run -p 8000:8000 customer-churn-api
```

## AWS Elastic Beanstalk Deployment

### Prerequisites
- AWS CLI installed and configured (`aws configure`)
- EB CLI installed (`pip install awsebcli`)

### Deploy

```bash
cd backend

# Initialize EB application (first time only)
eb init -p docker customer-churn-api --region us-east-1

# Create environment and deploy
eb create customer-churn-env --single --instance-type t3.small

# For subsequent deployments
eb deploy
```

### S3 Model Artifact Storage

Set environment variables to enable S3 upload during training:

```bash
export AWS_S3_BUCKET=your-bucket-name
export AWS_REGION=us-east-1
python app.py
```

Or configure in Elastic Beanstalk:
```bash
eb setenv AWS_S3_BUCKET=your-bucket-name AWS_REGION=us-east-1
```

## Dataset Columns

| Column | Type | Description |
|---|---|---|
| customerID | string | Unique ID (dropped during training) |
| gender | string | Male / Female |
| SeniorCitizen | int | 0 or 1 |
| Partner | string | Yes / No |
| Dependents | string | Yes / No |
| tenure | int | Months with company |
| PhoneService | string | Yes / No |
| MultipleLines | string | Yes / No / No phone service |
| InternetService | string | DSL / Fiber optic / No |
| OnlineSecurity | string | Yes / No / No internet service |
| OnlineBackup | string | Yes / No / No internet service |
| DeviceProtection | string | Yes / No / No internet service |
| TechSupport | string | Yes / No / No internet service |
| StreamingTV | string | Yes / No / No internet service |
| StreamingMovies | string | Yes / No / No internet service |
| Contract | string | Month-to-month / One year / Two year |
| PaperlessBilling | string | Yes / No |
| PaymentMethod | string | Electronic check / Mailed check / Bank transfer / Credit card |
| MonthlyCharges | float | Monthly charge amount |
| TotalCharges | float | Total charges to date |
| Churn | string | Yes / No (target variable) |

## Preprocessing Details

All preprocessing is done manually with Pandas (no sklearn Pipeline or ColumnTransformer):

1. **Drop** `customerID`
2. **Convert** `TotalCharges` to numeric, fill missing with median
3. **Encode target**: `Churn` → `Yes=1, No=0`
4. **Binary encode**: `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`
5. **Normalize service columns**: Replace "No internet service" / "No phone service" with "No", then binary encode
6. **One-hot encode**: `InternetService`, `Contract`, `PaymentMethod` (with `drop_first=True`)
7. **Save feature columns** to `feature_columns.json` for prediction alignment
