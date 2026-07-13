import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# AWS Credentials and S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY") or os.getenv("AWS_SECRET_KEY_ID") or os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME") or os.getenv("AWS_S3_BUCKET")

def print_config():
    """Print configuration for debugging (hiding secret values)."""
    print("--- S3 Configuration ---")
    print(f"AWS_REGION: {AWS_REGION}")
    print(f"BUCKET_NAME: {BUCKET_NAME}")
    print(f"AWS_ACCESS_KEY set: {bool(AWS_ACCESS_KEY)}")
    print(f"AWS_SECRET_KEY set: {bool(AWS_SECRET_KEY)}")
    print("------------------------")
