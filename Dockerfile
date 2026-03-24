# Base image
FROM python:3.10-slim

# Accept Run ID as a build argument
ARG RUN_ID

# Set working directory
WORKDIR /app

# Install MLflow (would be used to pull model in a real scenario)
RUN pip install --no-cache-dir mlflow scikit-learn

# Simulate downloading the model from the MLflow tracking server
CMD echo "Downloading model for Run ID: ${RUN_ID}" && \
    echo "Model download complete. Starting server..." && \
    echo "App running."
