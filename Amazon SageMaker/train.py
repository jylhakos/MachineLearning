#!/usr/bin/env python3
"""
SageMaker training script for Fish weight prediction
===================================================

This script is designed to run in Amazon SageMaker for training
a regression model to predict fish weight based on species and dimensions.

Author: Fish ML Project
Date: 2025
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json


def model_fn(model_dir):
    """Load model for SageMaker inference."""
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type='application/json'):
    """Parse input data for SageMaker inference."""
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        return pd.DataFrame(input_data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Make predictions using the loaded model."""
    predictions = model.predict(input_data)
    return predictions.tolist()


def output_fn(predictions, content_type='application/json'):
    """Format output for SageMaker inference."""
    if content_type == 'application/json':
        return json.dumps({'predictions': predictions})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def load_data(data_dir):
    """Load and prepare the fish dataset."""
    data_path = os.path.join(data_dir, 'Fish.csv')
    
    if not os.path.exists(data_path):
        # Alternative paths for different environments
        alternative_paths = [
            'Dataset/Fish.csv',
            '/opt/ml/input/data/training/Fish.csv',
            'Fish.csv'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            raise FileNotFoundError(f"Fish.csv not found in {data_dir} or alternative locations")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df


def preprocess_data(df):
    """Preprocess the data for training."""
    # Check for missing values
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    # Remove any rows with missing target variable
    df = df.dropna(subset=['Weight'])
    
    # Features and target
    feature_columns = ['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    X = df[feature_columns]
    y = df['Weight']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Species distribution:\n{X['Species'].value_counts()}")
    
    return X, y


def create_preprocessing_pipeline():
    """Create preprocessing pipeline for features."""
    # Define which columns need encoding
    categorical_features = ['Species']
    numerical_features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )
    
    return preprocessor


def train_model(X, y, model_type='linear_regression'):
    """Train the regression model."""
    print(f"\nTraining {model_type} model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X['Species']
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Choose the model based on model_type
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=1.0)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Print results
    print(f"\nTraining Results:")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # Create metrics dictionary
    metrics = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'model_type': model_type
    }
    
    return pipeline, metrics


def save_model(model, model_dir, metrics):
    """Save the trained model and metrics."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './'))
    parser.add_argument('--model-type', type=str, default='linear_regression',
                       choices=['linear_regression', 'ridge', 'lasso', 'random_forest'])
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FISH WEIGHT PREDICTION - SAGEMAKER TRAINING")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Training data directory: {args.train}")
    print(f"Model type: {args.model_type}")
    
    try:
        # Load and preprocess data
        df = load_data(args.train)
        X, y = preprocess_data(df)
        
        # Train model
        model, metrics = train_model(X, y, args.model_type)
        
        # Save model and metrics
        save_model(model, args.model_dir, metrics)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
