# Makefile for Fish ML project
# ============================
# This Makefile provides commands for common tasks

.PHONY: help setup test clean train serve docker-build docker-run aws-setup deploy

# Default target
help:
	@echo "🐟 Fish ML Project - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup and Development:"
	@echo "  make setup         - Set up Python environment and install dependencies"
	@echo "  make test          - Run test suite to validate setup"
	@echo "  make clean         - Clean up generated files and directories"
	@echo ""
	@echo "Local Development:"
	@echo "  make train         - Train model locally"
	@echo "  make serve         - Start FastAPI inference server"
	@echo "  make analysis      - Run data analysis"
	@echo "  make dev           - Start interactive development tool"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make docker-test   - Test Docker container"
	@echo ""
	@echo "AWS Deployment:"
	@echo "  make aws-setup     - Setup AWS infrastructure"
	@echo "  make deploy        - Deploy to AWS SageMaker"
	@echo ""
	@echo "Utilities:"
	@echo "  make requirements  - Generate requirements.txt from current environment"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format Python code"

# Setup Python environment
setup:
	@echo "🔧 Setting up Python environment..."
	chmod +x setup.sh
	./setup.sh

# Run tests
test:
	@echo "🧪 Running test suite..."
	@if [ -d "venv" ]; then \
		source venv/bin/activate && python test_setup.py; \
	else \
		python3 test_setup.py; \
	fi

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -rf venv/
	rm -rf model/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete
	@echo "✅ Cleanup complete"

# Train model locally
train:
	@echo "🏋️  Training model locally..."
	@if [ -d "venv" ]; then \
		source venv/bin/activate && python train.py --model-dir ./model --train ./; \
	else \
		python3 train.py --model-dir ./model --train ./; \
	fi

# Start inference server
serve:
	@echo "🚀 Starting inference server..."
	@echo "   Access API at: http://localhost:8000"
	@echo "   API docs at: http://localhost:8000/docs"
	@if [ -d "venv" ]; then \
		source venv/bin/activate && python inference_server.py; \
	else \
		python3 inference_server.py; \
	fi

# Run data analysis
analysis:
	@echo "📊 Running data analysis..."
	@if [ -d "venv" ]; then \
		source venv/bin/activate && python fish_analysis.py; \
	else \
		python3 fish_analysis.py; \
	fi

# Start development tool
dev:
	@echo "🛠️  Starting development tool..."
	@if [ -d "venv" ]; then \
		source venv/bin/activate && python local_dev.py; \
	else \
		python3 local_dev.py; \
	fi

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t fish-ml .
	@echo "✅ Docker image built: fish-ml"

# Run Docker container
docker-run:
	@echo "🐳 Running Docker container..."
	@echo "   Starting inference server on port 8000..."
	docker run -p 8000:8000 -v $(PWD)/model:/opt/ml/model fish-ml

# Test Docker container
docker-test:
	@echo "🧪 Testing Docker container..."
	docker run --rm -v $(PWD)/Dataset:/opt/ml/input/data/training fish-ml train --model-type linear_regression

# Setup AWS infrastructure
aws-setup:
	@echo "☁️  Setting up AWS infrastructure..."
	@if [ -d "venv" ]; then \
		source venv/bin/activate && python aws_setup.py; \
	else \
		python3 aws_setup.py; \
	fi

# Deploy to AWS SageMaker
deploy:
	@echo "🚀 Deploying to AWS SageMaker..."
	@if [ -d "venv" ]; then \
		source venv/bin/activate && python deploy_sagemaker.py; \
	else \
		python3 deploy_sagemaker.py; \
	fi

# Generate requirements.txt
requirements:
	@echo "📋 Generating requirements.txt..."
	@if [ -d "venv" ]; then \
		source venv/bin/activate && pip freeze > requirements_generated.txt; \
		echo "✅ Generated requirements_generated.txt"; \
	else \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
	fi

# Lint code (if tools are available)
lint:
	@echo "🔍 Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 *.py; \
	else \
		echo "⚠️  flake8 not found, skipping linting"; \
	fi
	@if command -v pylint >/dev/null 2>&1; then \
		pylint *.py; \
	else \
		echo "⚠️  pylint not found, skipping linting"; \
	fi

# Format code (if tools are available)
format:
	@echo "✨ Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black *.py; \
	else \
		echo "⚠️  black not found, skipping formatting"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort *.py; \
	else \
		echo "⚠️  isort not found, skipping import sorting"; \
	fi

# Quick start - setup and train
quick-start: setup train
	@echo "🎉 Quick start complete!"
	@echo "   Model trained and ready for inference"
	@echo "   Run 'make serve' to start the API server"

# Full pipeline - setup, train, test, build
full-pipeline: setup test train docker-build
	@echo "🎉 Full pipeline complete!"
	@echo "   Ready for AWS deployment with 'make deploy'"
