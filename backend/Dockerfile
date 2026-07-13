# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY preprocess.py .
COPY main.py .
COPY app.py .
COPY config.py .

# Copy model artifacts (pre-trained model and feature columns)
# These should exist after running: python app.py
COPY models/ ./models/

# Copy dataset (needed if retraining inside container)
COPY data/ ./data/

# Expose port 8000 (FastAPI default)
# AWS Elastic Beanstalk uses the PORT environment variable
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI with Uvicorn
# Use $PORT for AWS EB compatibility, default to 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
