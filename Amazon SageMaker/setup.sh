#!/bin/bash

# Fish ML setup script
# ============================
# This script sets up the complete Python environment for the Fish ML project
# including all dependencies for local development and AWS deployment.

echo "🐟 Fish Machine Learning Project Setup"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "Dataset/Fish.csv" ]; then
    echo "❌ Error: Fish.csv not found in Dataset/ folder"
    echo "Please ensure you're in the project root directory"
    exit 1
fi

echo "✅ Dataset found: Dataset/Fish.csv"

# Check Python version
echo ""
echo "🐍 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or later"
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists, removing old one..."
    rm -rf venv
fi

python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created successfully"
else
    echo "❌ Failed to create virtual environment"
    echo "Please ensure Python 3 venv module is available"
    exit 1
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo ""
echo "📋 Installing required packages..."
echo "This may take a few minutes..."

# Install core ML packages first
echo "   Installing core ML packages..."
pip install scikit-learn pandas numpy matplotlib seaborn jupyter joblib scipy plotly

# Install AWS packages
echo "   Installing AWS packages..."
pip install boto3 sagemaker awscli

# Install FastAPI and web service packages  
echo "   Installing API framework..."
pip install fastapi "uvicorn[standard]" pydantic

# Install from requirements.txt to ensure all versions match
echo "   Installing from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully"
else
    echo "❌ Failed to install some packages"
    echo "You may need to install them manually"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 -c "
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import matplotlib.pyplot as plt
    import seaborn as sns
    import fastapi
    import uvicorn
    import joblib
    print('✅ Core ML packages: OK')
    
    try:
        import boto3
        import sagemaker
        print('✅ AWS packages: OK')
    except ImportError as e:
        print(f'⚠️  AWS packages: {e}')
        print('   (AWS packages are optional for local development)')
    
    print('✅ All essential packages verified!')
    
except ImportError as e:
    print(f'❌ Package verification failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Package verification failed"
    exit 1
fi

# Create model directory
echo ""
echo "📁 Creating model directory..."
mkdir -p model

# Make scripts executable  
echo ""
echo "🔧 Making scripts executable..."
chmod +x train.py
chmod +x inference_server.py
chmod +x local_dev.py
chmod +x deploy_sagemaker.py
chmod +x aws_setup.py
chmod +x docker-entrypoint.sh

# Display next steps
echo ""
echo "🎉 Setup completed successfully!"
echo "========================================="
echo ""
echo "📋 What's available:"
echo "   📊 fish_analysis.py      - Comprehensive data analysis"
echo "   🏋️  fish_regression.py    - Basic regression model"
echo "   🤖 train.py             - SageMaker training script"
echo "   🚀 inference_server.py   - FastAPI inference server"
echo "   🛠️  local_dev.py         - Interactive development tool"
echo "   ☁️  deploy_sagemaker.py  - AWS deployment script"
echo "   🔧 aws_setup.py         - AWS infrastructure setup"
echo ""
echo "🚀 Quick start options:"
echo ""
echo "   1. Run interactive development tool:"
echo "      python local_dev.py"
echo ""
echo "   2. Train model locally:"
echo "      source venv/bin/activate"
echo "      python train.py --model-dir ./model --train ./"
echo ""
echo "   3. Start inference server:"
echo "      source venv/bin/activate" 
echo "      python inference_server.py"
echo "      # Then visit: http://localhost:8000/docs"
echo ""
echo "   4. Run data analysis:"
echo "      source venv/bin/activate"
echo "      python fish_analysis.py"
echo ""
echo "   5. Setup AWS (for cloud deployment):"
echo "      source venv/bin/activate"
echo "      python aws_setup.py"
echo ""
echo "📖 See README.md for detailed documentation"
echo ""
echo "🐟 Happy fish weight predicting!"

echo ""
echo "🎯 Setup complete! Here's what you can do next:"
echo ""
echo "1. Run regression analysis:"
echo "   python fish_regression.py"
echo ""
echo "2. Run classification analysis:"
echo "   python fish_classification.py"
echo ""
echo "3. Run comprehensive analysis:"
echo "   python fish_analysis.py"
echo ""
echo "4. Start Jupyter notebook for interactive analysis:"
echo "   jupyter notebook"
echo ""
echo "📝 Don't forget to activate your virtual environment each time:"
echo "   source venv/bin/activate"
echo ""
echo "🎉 Happy fish machine learning!"
