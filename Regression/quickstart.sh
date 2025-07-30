#!/bin/bash

# Script to start Fish ML
# ====================================

echo " Fish Machine Learning - Quick Start Guide"
echo "============================================="

# Check current directory
if [ ! -f "Dataset/Fish.csv" ]; then
    echo "❌ Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: Dataset/Fish.csv, README.md, requirements.txt"
    exit 1
fi

echo " Found project files in current directory"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo " Virtual environment not found. Creating one..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo " Virtual environment created"
    else
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo ""
echo " Activating virtual environment..."
source venv/bin/activate

# Check if packages are installed
echo ""
echo " Checking if required packages are installed..."
python3 -c "
try:
    import pandas, sklearn, numpy, matplotlib
    print(' All packages are already installed')
    installed = True
except ImportError as e:
    print(' Some packages missing:', str(e))
    installed = False
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo " Installing required packages..."
    pip install -q scikit-learn pandas numpy matplotlib seaborn
    
    if [ $? -eq 0 ]; then
        echo " Packages installed successfully"
    else
        echo "❌ Failed to install packages"
        exit 1
    fi
fi

echo ""
echo "Setup complete"
echo ""
echo "1️⃣  Basic regression example (Linear Regression for weight prediction):"
echo "   python3 fish_regression.py"
echo ""
echo "2️⃣  Classification example (KNN for species prediction):"
echo "   python3 fish_classification.py"
echo ""
echo "3️⃣  Comprehensive analysis (Multiple models and visualizations):"
echo "   python3 fish_analysis.py"
echo ""
echo "4️⃣  Verify dataset only:"
echo "   python3 verify_dataset.py"
echo ""
echo "5️⃣  Open Jupyter notebook for interactive analysis:"
echo "   jupyter notebook"
echo ""

# Interactive menu
echo "Enter your choice (1-5), or press Enter to see project info:"
read -r choice

case $choice in
    1)
        echo " Running regression analysis..."
        python3 fish_regression.py
        ;;
    2)
        echo " Running classification analysis..."
        python3 fish_classification.py
        ;;
    3)
        echo " Running comprehensive analysis..."
        python3 fish_analysis.py
        ;;
    4)
        echo " Verifying dataset..."
        python3 verify_dataset.py
        ;;
    5)
        echo " Starting Jupyter Notebook..."
        jupyter notebook
        ;;
    *)
        echo ""
        echo " Project Information:"
        echo "======================"
        echo "📁 Project structure:"
        find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.csv" -o -name "*.sh" | sort
        echo ""
        echo " Read the full documentation:"
        echo "   cat README.md"
        echo ""
        echo " To run this quick start again:"
        echo "   ./quickstart.sh"
        ;;
esac

echo ""
echo "Always activate your virtual environment with 'source venv/bin/activate'"
