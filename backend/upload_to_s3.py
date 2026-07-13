import os
import sys
import boto3
from pathlib import Path

# Add parent directory to sys.path so config can be imported if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

# Required model artifact files
REQUIRED_FILES = [
    "MLmodel",
    "model.skops",
    "conda.yaml",
    "python_env.yaml",
    "requirements.txt"
]

def main():
    print("====================================================")
    print("  AWS S3 MODEL ARTIFACT UPLOADER")
    print("====================================================\n")
    
    # Print configuration status
    config.print_config()
    
    # Verify .env values are set
    if not config.AWS_ACCESS_KEY or not config.AWS_SECRET_KEY or not config.BUCKET_NAME:
        print("[ERROR] AWS credentials or BUCKET_NAME not set in backend/.env!")
        print("Please edit backend/.env and populate it with your AWS credentials.")
        sys.exit(1)
        
    # Check local registered_model/ directory
    registered_model_dir = Path(__file__).resolve().parent / "registered_model"
    if not registered_model_dir.exists():
        print(f"[ERROR] Local model folder '{registered_model_dir}' not found!")
        print("Please make sure you have copied the registered model artifacts locally.")
        sys.exit(1)
        
    # Check if all required files exist
    missing_files = []
    for f in REQUIRED_FILES:
        f_path = registered_model_dir / f
        if not f_path.exists():
            missing_files.append(f)
            
    if missing_files:
        print(f"[ERROR] Missing required files in registered_model/: {missing_files}")
        sys.exit(1)
        
    print("[Validation] All required model artifacts found locally. Proceeding to upload...\n")
    
    try:
        # Initialize boto3 S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_SECRET_KEY,
            region_name=config.AWS_REGION
        )
        
        # Upload each file
        for f in REQUIRED_FILES:
            local_file_path = registered_model_dir / f
            s3_key = f"registered_model/{f}"
            print(f"Uploading: {f} -> s3://{config.BUCKET_NAME}/{s3_key} ...")
            s3_client.upload_file(
                str(local_file_path),
                config.BUCKET_NAME,
                s3_key
            )
            print(f"  [OK] Uploaded {f}")
            
        print("\n====================================================")
        print("  UPLOAD COMPLETED SUCCESSFULLY!")
        print("====================================================")
        
    except Exception as e:
        print(f"\n[ERROR] S3 upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
