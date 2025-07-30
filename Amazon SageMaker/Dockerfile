# Dockerfile for Fish weight prediction ML model
# ============================================
# This Dockerfile creates a container for both training and inference
# of the fish weight prediction model, suitable for AWS SageMaker deployment.

FROM python:3.11-slim

# Set working directory
WORKDIR /opt/ml

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/ml/code:${PATH}"

# Create necessary directories for SageMaker
RUN mkdir -p /opt/ml/input/data/training \
    /opt/ml/input/config \
    /opt/ml/model \
    /opt/ml/output \
    /opt/ml/code

# Copy requirements first for better caching
COPY requirements.txt /opt/ml/code/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copy application code
COPY train.py /opt/ml/code/
COPY inference_server.py /opt/ml/code/
COPY fish_analysis.py /opt/ml/code/
COPY fish_regression.py /opt/ml/code/
COPY Dataset/ /opt/ml/code/Dataset/

# Make scripts executable
RUN chmod +x /opt/ml/code/train.py /opt/ml/code/inference_server.py

# SageMaker training entrypoint
COPY docker-entrypoint.sh /opt/ml/code/
RUN chmod +x /opt/ml/code/docker-entrypoint.sh

# Expose port for inference server
EXPOSE 8000 8080

# Default command (can be overridden)
CMD ["/opt/ml/code/docker-entrypoint.sh"]
