#!/usr/bin/env python3
"""
Demo script for Fish ML project
====================================

This script demonstrates the key features of the Fish ML project,
including model training, prediction, and API usage.

Author: Fish ML Project
Date: 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append('.')

def demo_data_analysis():
    """Demonstrate data loading and basic analysis."""
    print("🐟 Fish ML Project Demo")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv('Dataset/Fish.csv')
    
    print("📊 Dataset Overview:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Species: {df['Species'].nunique()}")
    
    print("\n📈 Sample Data:")
    print(df.head(3).to_string(index=False))
    
    print("\n📊 Statistics:")
    print(f"   Average Weight: {df['Weight'].mean():.1f}g")
    print(f"   Weight Range: {df['Weight'].min():.1f}g - {df['Weight'].max():.1f}g")
    
    print("\n🐟 Species Distribution:")
    species_counts = df['Species'].value_counts()
    for species, count in species_counts.head(5).items():
        print(f"   {species}: {count} fish")


def demo_model_training():
    """Demonstrate model training and evaluation."""
    print("\n🏋️  Model Training Demo")
    print("=" * 50)
    
    # Check if model exists
    if os.path.exists('model/model.joblib') and os.path.exists('model/metrics.json'):
        print("✅ Trained model found!")
        
        # Load metrics
        with open('model/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print("📊 Model Performance:")
        print(f"   Model Type: {metrics['model_type']}")
        print(f"   R² Score: {metrics['test_r2']:.4f}")
        print(f"   Mean Absolute Error: {metrics['test_mae']:.1f}g")
        print(f"   Root Mean Squared Error: {np.sqrt(metrics['test_mse']):.1f}g")
        
    else:
        print("❌ No trained model found. Run 'python train.py' first.")


def demo_prediction():
    """Demonstrate making predictions with the trained model."""
    print("\n🔮 Prediction Demo")
    print("=" * 50)
    
    try:
        import joblib
        
        # Load the trained model
        model = joblib.load('model/model.joblib')
        
        # Sample fish data
        sample_fish = [
            {"Species": "Bream", "Length1": 23.2, "Length2": 25.4, "Length3": 30.0, "Height": 11.52, "Width": 4.02},
            {"Species": "Pike", "Length1": 37.0, "Length2": 40.0, "Length3": 43.5, "Height": 13.8, "Width": 5.1},
            {"Species": "Perch", "Length1": 16.2, "Length2": 18.0, "Length3": 19.8, "Height": 8.9, "Width": 3.2}
        ]
        
        print("🐟 Sample Predictions:")
        
        for i, fish in enumerate(sample_fish, 1):
            # Convert to DataFrame
            df_fish = pd.DataFrame([fish])
            
            # Make prediction
            predicted_weight = model.predict(df_fish)[0]
            
            print(f"\n   Fish #{i} ({fish['Species']}):")
            print(f"   Dimensions: L1={fish['Length1']}, L2={fish['Length2']}, L3={fish['Length3']}")
            print(f"   H={fish['Height']}, W={fish['Width']}")
            print(f"   🎯 Predicted Weight: {predicted_weight:.1f}g")
            
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        print("   Make sure the model is trained first.")


def demo_api_usage():
    """Demonstrate API usage examples."""
    print("\n🚀 API Usage Demo")
    print("=" * 50)
    
    print("📝 Example API calls:")
    
    # Health check example
    print("\n1. Health Check:")
    print("   curl http://localhost:8000/health")
    
    # Single prediction example
    print("\n2. Single Prediction:")
    print("   curl -X POST 'http://localhost:8000/predict' \\")
    print("   -H 'Content-Type: application/json' \\")
    print("   -d '{")
    print("     \"species\": \"Bream\",")
    print("     \"length1\": 23.2,")
    print("     \"length2\": 25.4,")
    print("     \"length3\": 30.0,")
    print("     \"height\": 11.52,")
    print("     \"width\": 4.02")
    print("   }'")
    
    # Batch prediction example
    print("\n3. Batch Prediction:")
    print("   curl -X POST 'http://localhost:8000/predict/batch' \\")
    print("   -H 'Content-Type: application/json' \\")
    print("   -d '{")
    print("     \"fish_data\": [")
    print("       {\"species\": \"Bream\", \"length1\": 23.2, \"length2\": 25.4, \"length3\": 30.0, \"height\": 11.52, \"width\": 4.02},")
    print("       {\"species\": \"Pike\", \"length1\": 37.0, \"length2\": 40.0, \"length3\": 43.5, \"height\": 13.8, \"width\": 5.1}")
    print("     ]")
    print("   }'")


def demo_deployment_options():
    """Show deployment options."""
    print("\n☁️  Deployment Options")
    print("=" * 50)
    
    print("🐳 Docker Deployment:")
    print("   docker build -t fish-ml .")
    print("   docker run -p 8000:8000 -v $(pwd)/model:/opt/ml/model fish-ml")
    
    print("\n☁️  AWS SageMaker Deployment:")
    print("   python aws_setup.py       # Setup AWS infrastructure")
    print("   python deploy_sagemaker.py  # Deploy to SageMaker")
    
    print("\n🛠️  Local Development:")
    print("   python local_dev.py       # Interactive development tool")
    print("   make setup                # Alternative setup using Make")
    print("   make train                # Train model using Make")
    print("   make serve                # Start API server using Make")


def demo_project_structure():
    """Show project structure and files."""
    print("\n📁 Project Structure")
    print("=" * 50)
    
    structure = {
        "Dataset/": "Training data (Fish.csv)",
        "model/": "Trained model artifacts (generated)",
        "train.py": "SageMaker training script",
        "inference_server.py": "FastAPI inference server",
        "fish_analysis.py": "Comprehensive data analysis",
        "fish_regression.py": "Basic regression example",
        "local_dev.py": "Interactive development tool",
        "deploy_sagemaker.py": "AWS SageMaker deployment",
        "aws_setup.py": "AWS infrastructure setup",
        "Dockerfile": "Container configuration",
        "requirements.txt": "Python dependencies",
        "README.md": "Complete documentation"
    }
    
    for file_path, description in structure.items():
        print(f"   {file_path:<20} {description}")


def main():
    """Run the complete demo."""
    try:
        demo_data_analysis()
        demo_model_training()
        demo_prediction()
        demo_api_usage()
        demo_deployment_options()
        demo_project_structure()
        
        print("\n" + "=" * 50)
        print("🎉 Demo Completed Successfully!")
        print("=" * 50)
        
        print("\n🚀 Next Steps:")
        print("   1. Explore the data: python fish_analysis.py")
        print("   2. Start API server: python inference_server.py")
        print("   3. Try different models: python train.py --model-type ridge")
        print("   4. Use development tool: python local_dev.py")
        print("   5. Deploy to AWS: python aws_setup.py")
        
        print("\n📖 Full documentation available in README.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
