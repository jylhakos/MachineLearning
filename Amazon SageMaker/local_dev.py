#!/usr/bin/env python3
"""
Local development and testing script
===================================

This script helps with local development, testing, and Docker operations
before deploying to AWS SageMaker.

Author: Fish ML Project
Date: 2025
"""

import os
import subprocess
import sys
import json
import requests
import time
from pathlib import Path


def run_command(command, description="", check=True):
    """Run a shell command and print the result."""
    if description:
        print(f"🔄 {description}")
    
    print(f"   Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        print(f"❌ Error: {result.stderr}")
        raise Exception(f"Command failed: {command}")
    
    if result.stdout:
        print(f"   Output: {result.stdout.strip()}")
    
    return result


def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("🐍 Setting up Python environment...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        run_command('python3 -m venv venv', "Creating virtual environment")
    
    # Install requirements
    if os.name == 'nt':  # Windows
        activate_cmd = 'venv\\Scripts\\activate && '
    else:  # Unix/Linux
        activate_cmd = 'source venv/bin/activate && '
    
    run_command(f'{activate_cmd}pip install --upgrade pip', "Upgrading pip")
    run_command(f'{activate_cmd}pip install -r requirements.txt', "Installing dependencies")
    
    print("✅ Python environment ready")


def train_model_locally():
    """Train the model locally."""
    print("🏋️  Training model locally...")
    
    if os.name == 'nt':  # Windows
        activate_cmd = 'venv\\Scripts\\activate && '
    else:  # Unix/Linux
        activate_cmd = 'source venv/bin/activate && '
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Train the model
    run_command(
        f'{activate_cmd}python train.py --model-dir ./model --train ./',
        "Training linear regression model"
    )
    
    # Check if model was created
    if os.path.exists('model/model.joblib'):
        print("✅ Model trained successfully")
        return True
    else:
        print("❌ Model training failed - no model file found")
        return False


def start_inference_server():
    """Start the FastAPI inference server locally."""
    print("🚀 Starting local inference server...")
    
    if os.name == 'nt':  # Windows
        activate_cmd = 'venv\\Scripts\\activate && '
    else:  # Unix/Linux
        activate_cmd = 'source venv/bin/activate && '
    
    # Start server in background
    command = f'{activate_cmd}uvicorn inference_server:app --host 0.0.0.0 --port 8000 --reload'
    print(f"   Starting server with: {command}")
    print("   Access the API at: http://localhost:8000")
    print("   API documentation at: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop the server")
    
    subprocess.run(command, shell=True)


def test_inference_server():
    """Test the local inference server."""
    print("🧪 Testing local inference server...")
    
    base_url = "http://localhost:8000"
    
    # Wait for server to start
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Server is running")
                break
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"   Waiting for server... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("❌ Server is not responding")
                return False
    
    # Test prediction endpoint
    test_data = {
        "species": "Bream",
        "length1": 23.2,
        "length2": 25.4,
        "length3": 30.0,
        "height": 11.52,
        "width": 4.02
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful: {result['predicted_weight']:.2f}g")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing prediction: {str(e)}")
        return False


def build_docker_image():
    """Build Docker image locally."""
    print("🐳 Building Docker image...")
    
    # Build the image
    run_command(
        'docker build -t fish-ml-local .',
        "Building Docker image"
    )
    
    print("✅ Docker image built successfully")


def test_docker_container():
    """Test the Docker container locally."""
    print("🧪 Testing Docker container...")
    
    # Stop any existing container
    run_command('docker stop fish-ml-test 2>/dev/null || true', check=False)
    run_command('docker rm fish-ml-test 2>/dev/null || true', check=False)
    
    # Run training in container
    print("   Testing training mode...")
    run_command(
        'docker run --name fish-ml-test -v $(pwd)/Dataset:/opt/ml/input/data/training fish-ml-local train',
        "Running training in container"
    )
    
    # Clean up
    run_command('docker rm fish-ml-test', check=False)
    
    # Test inference server
    print("   Testing inference server mode...")
    print("   Starting container on port 8001...")
    
    # Run inference server in background
    result = subprocess.Popen([
        'docker', 'run', '--rm', '-p', '8001:8000', 
        '-v', f'{os.getcwd()}/model:/opt/ml/model',
        'fish-ml-local'
    ])
    
    # Wait a bit for server to start
    time.sleep(5)
    
    # Test the containerized server
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            print("✅ Docker container server is working")
            success = True
        else:
            print("❌ Docker container server failed")
            success = False
    except Exception as e:
        print(f"❌ Error testing Docker container: {str(e)}")
        success = False
    finally:
        # Stop the container
        result.terminate()
        result.wait()
    
    return success


def run_analysis():
    """Run the fish analysis script."""
    print("📊 Running fish data analysis...")
    
    if os.name == 'nt':  # Windows
        activate_cmd = 'venv\\Scripts\\activate && '
    else:  # Unix/Linux
        activate_cmd = 'source venv/bin/activate && '
    
    run_command(
        f'{activate_cmd}python fish_analysis.py',
        "Running comprehensive fish analysis"
    )


def display_menu():
    """Display the main menu."""
    print("\n🐟 Fish ML - Local Development Menu")
    print("=" * 50)
    print("1. Setup Python environment")
    print("2. Train model locally")
    print("3. Start inference server")
    print("4. Test inference server")
    print("5. Run data analysis")
    print("6. Build Docker image")
    print("7. Test Docker container")
    print("8. Full local pipeline")
    print("9. Exit")
    print("=" * 50)


def full_pipeline():
    """Run the complete local development pipeline."""
    print("🚀 Running full local development pipeline...")
    
    steps = [
        ("Setup environment", setup_python_environment),
        ("Train model", train_model_locally),
        ("Build Docker image", build_docker_image),
        ("Test Docker container", test_docker_container),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        try:
            success = step_func()
            if success is False:
                print(f"❌ Pipeline failed at step: {step_name}")
                return False
        except Exception as e:
            print(f"❌ Pipeline failed at step '{step_name}': {str(e)}")
            return False
    
    print("\n🎉 Full pipeline completed successfully!")
    return True


def main():
    """Main function."""
    print("🐟 Fish ML - Local Development Tools")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('Dataset/Fish.csv'):
        print("❌ Error: Fish.csv not found in Dataset/ folder")
        print("Please ensure you're in the project root directory")
        return 1
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == '1':
                setup_python_environment()
            elif choice == '2':
                train_model_locally()
            elif choice == '3':
                start_inference_server()
            elif choice == '4':
                test_inference_server()
            elif choice == '5':
                run_analysis()
            elif choice == '6':
                build_docker_image()
            elif choice == '7':
                test_docker_container()
            elif choice == '8':
                full_pipeline()
            elif choice == '9':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-9.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return 0


if __name__ == '__main__':
    exit(main())
