FROM python:3.8-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Set up MLFlow tracking server
RUN mkdir -p /mlflow/artifacts
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_ARTIFACT_ROOT=file:///mlflow/artifacts

# Copy pipeline code
COPY . /app

# Run pipeline
CMD ["python", "DAG.py"]
