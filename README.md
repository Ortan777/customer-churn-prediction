# Customer Churn Prediction using MLOps

An end-to-end Machine Learning project that predicts customer churn using the IBM Telco Customer Churn dataset. The project demonstrates the complete MLOps workflow from model training to cloud deployment using MLflow, Amazon S3, Docker, FastAPI, and AWS Elastic Beanstalk.

---
Demo ="http://customer-churn-api-env.eba-523x2gmq.eu-north-1.elasticbeanstalk.com/docs#/"

## Project Overview

This project predicts whether a telecom customer is likely to churn based on their subscription details and usage patterns.

The workflow includes:

- Data preprocessing
- Feature engineering
- Model training using XGBoost
- MLflow experiment tracking
- Model registration
- Model artifact storage in Amazon S3
- FastAPI prediction API
- Docker containerization
- Deployment on AWS Elastic Beanstalk

---

## Architecture

```
IBM Telco Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Train XGBoost Model
        │
        ▼
MLflow Tracking
        │
        ▼
Register Model
        │
        ▼
Upload Model Artifacts to Amazon S3
        │
        ▼
FastAPI Backend
        │
        ▼
Automatically Download Model from S3
        │
        ▼
Docker Container
        │
        ▼
AWS Elastic Beanstalk
        │
        ▼
Prediction API
```

---

# Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Experiment Tracking | MLflow |
| Backend | FastAPI |
| Model Storage | Amazon S3 |
| Containerization | Docker |
| Cloud Deployment | AWS Elastic Beanstalk |
| Frontend | React + Vite |

---

# Project Structure

```
customer-churn-prediction/
│
├── backend/
│   ├── data/
│   ├── models/
│   ├── registered_model/
│   ├── app.py
│   ├── preprocess.py
│   ├── main.py
│   ├── config.py
│   ├── upload_to_s3.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── Dockerrun.aws.json
│
├── frontend/
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

# Features

- Predict customer churn using XGBoost
- Data preprocessing and feature engineering
- MLflow experiment tracking
- Model registration
- Automatic model download from Amazon S3
- REST API using FastAPI
- Dockerized application
- AWS Elastic Beanstalk deployment
- React dashboard for predictions

---

# Dataset

Dataset used:

**IBM Telco Customer Churn Dataset**

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract
- Payment Method
- Monthly Charges
- Total Charges
- Churn (Target)

---

# Installation

Clone the repository

```bash
git clone https://github.com/Ortan777/customer-churn-prediction.git

cd customer-churn-prediction
```

---

## Backend Setup

```bash
cd backend

pip install -r requirements.txt
```

Create a `.env` file

```env
AWS_ACCESS_KEY=YOUR_ACCESS_KEY
AWS_SECRET_KEY=YOUR_SECRET_KEY
AWS_REGION=YOUR_REGION
BUCKET_NAME=YOUR_BUCKET_NAME
```

---

## Train the Model

```bash
python app.py
```

This will:

- preprocess the dataset
- train Logistic Regression and XGBoost
- compare model performance
- log experiments in MLflow
- register the best model
- upload model artifacts to Amazon S3

---

## Run the API

```bash
uvicorn main:app --reload
```

API will be available at

```
http://localhost:8000
```

Swagger UI

```
http://localhost:8000/docs
```

---

# API Endpoints

## Health Check

```
GET /health
```

---

## Predict Customer Churn

```
POST /predict
```

Example Request

```json
{
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
  "TotalCharges": 844.20
}
```

Example Response

```json
{
    "prediction": "Yes",
    "probability": 0.82
}
```

---

# MLflow

Start MLflow UI

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

---

# Docker

Build the image

```bash
docker build -t customer-churn-api .
```

Run the container

```bash
docker run -p 8000:8000 --env-file .env customer-churn-api
```

---

# AWS Deployment

The backend is deployed using:

- Docker
- AWS Elastic Beanstalk
- Amazon S3

Deployment flow

```
Train Model
      ↓
MLflow Registry
      ↓
Amazon S3
      ↓
FastAPI
      ↓
Docker
      ↓
Elastic Beanstalk
```

---

# Frontend

The project also includes a React frontend for interacting with the prediction API.

Run locally

```bash
cd frontend

npm install

npm run dev
```

---

# Future Improvements

- Deploy frontend with HTTPS support
- CI/CD using GitHub Actions
- Model monitoring
- Automated retraining
- Multiple model version support

---

# Author

**Ayush Shetty**

B.E. Computer Science (Data Science)

Alva's Institute of Engineering and Technology

GitHub: https://github.com/Ortan777

---

## License

This project is created for learning and academic purposes.
