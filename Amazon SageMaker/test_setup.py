#!/usr/bin/env python3
"""
Test Script for Fish ML project
==============================

This script runs basic tests to ensure the project is set up correctly
and all components are working as expected.

Author: Fish ML
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json


def test_data_loading():
    """Test if the Fish dataset can be loaded."""
    print("🧪 Testing data loading...")
    
    try:
        # Check if dataset exists
        data_path = 'Dataset/Fish.csv'
        if not os.path.exists(data_path):
            print(f"❌ Dataset not found at {data_path}")
            return False
        
        # Load dataset
        df = pd.read_csv(data_path)
        
        # Basic validation
        expected_columns = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
        missing_columns = set(expected_columns) - set(df.columns)
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        if len(df) == 0:
            print("❌ Dataset is empty")
            return False
        
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False


def test_model_training():
    """Test if model training works."""
    print("🧪 Testing model training...")
    
    try:
        # Import training module
        sys.path.append('.')
        from train import load_data, preprocess_data, train_model
        
        # Load and preprocess data
        df = load_data('./')
        X, y = preprocess_data(df)
        
        # Train a simple model
        model, metrics = train_model(X, y, 'linear_regression')
        
        # Validate metrics
        if metrics['test_r2'] < 0.5:
            print(f"⚠️  Low R² score: {metrics['test_r2']:.3f}")
        
        print(f"✅ Model training successful - R² Score: {metrics['test_r2']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        return False


def test_inference_server_import():
    """Test if inference server can be imported."""
    print("🧪 Testing inference server import...")
    
    try:
        from inference_server import FishMeasurements, predict_weight
        
        # Test Pydantic model
        sample_data = FishMeasurements(
            species="Bream",
            length1=23.2,
            length2=25.4,
            length3=30.0,
            height=11.52,
            width=4.02
        )
        
        print("✅ Inference server components imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error importing inference server: {str(e)}")
        return False


def test_package_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    packages = {
        'Core ML': ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'joblib'],
        'FastAPI': ['fastapi', 'uvicorn', 'pydantic'],
        'AWS (optional)': ['boto3', 'sagemaker']
    }
    
    all_success = True
    
    for category, package_list in packages.items():
        print(f"   Testing {category} packages...")
        category_success = True
        
        for package in package_list:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError as e:
                if category == 'AWS (optional)':
                    print(f"   ⚠️  {package} (optional)")
                else:
                    print(f"   ❌ {package}: {str(e)}")
                    category_success = False
                    all_success = False
        
        if category_success or category == 'AWS (optional)':
            print(f"   ✅ {category}: OK")
        else:
            print(f"   ❌ {category}: FAILED")
    
    return all_success


def test_file_structure():
    """Test if all required files are present."""
    print("🧪 Testing file structure...")
    
    required_files = [
        'Dataset/Fish.csv',
        'fish_analysis.py',
        'fish_regression.py',
        'train.py',
        'inference_server.py',
        'local_dev.py',
        'requirements.txt',
        'setup.sh',
        'README.md',
        'Dockerfile',
        'docker-entrypoint.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True


def test_python_version():
    """Test if Python version is compatible."""
    print("🧪 Testing Python version...")
    
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Minimum required: Python 3.8")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def test_write_permissions():
    """Test if we can write to the model directory."""
    print("🧪 Testing write permissions...")
    
    try:
        # Create model directory if it doesn't exist
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        
        # Test writing a file
        test_file = os.path.join(model_dir, 'test_file.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        # Clean up
        os.remove(test_file)
        
        print(f"✅ Write permissions OK for {model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Write permission error: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and return overall result."""
    print("🐟 Fish ML Project - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_package_imports),
        ("File Structure", test_file_structure),
        ("Write Permissions", test_write_permissions),
        ("Data Loading", test_data_loading),
        ("Model Training", test_model_training),
        ("Inference Server", test_inference_server_import),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


def main():
    """Main test function."""
    success = run_all_tests()
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Train a model: python train.py --model-dir ./model --train ./")
        print("   2. Start API server: python inference_server.py")
        print("   3. Run analysis: python fish_analysis.py")
        print("   4. Use development tool: python local_dev.py")
        return 0
    else:
        print("\n🔧 Please fix the failed tests before proceeding.")
        return 1


if __name__ == '__main__':
    exit(main())
