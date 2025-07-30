#!/bin/bash
# Docker entrypoint script for Fish ML container
# ==============================================
# This script handles both training and inference modes
# for AWS SageMaker deployment.

set -e

# Change to code directory
cd /opt/ml/code

echo "🐟 Fish ML Container Starting..."
echo "Current directory: $(pwd)"
echo "Container arguments: $@"

# Check if this is a training job (SageMaker training)
if [ "$1" = "train" ]; then
    echo "🏋️  Starting training mode..."
    exec python train.py "${@:2}"

# Check if this is serving mode (SageMaker inference)
elif [ "$1" = "serve" ]; then
    echo "🚀 Starting inference server..."
    exec python inference_server.py

# Default mode - start FastAPI server
else
    echo "🚀 Starting FastAPI inference server (default mode)..."
    exec uvicorn inference_server:app --host 0.0.0.0 --port 8000
fi
