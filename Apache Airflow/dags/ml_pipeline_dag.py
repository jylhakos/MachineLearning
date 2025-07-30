"""
Apache Airflow DAG for Fish weight prediction ML pipeline
=========================================================

This DAG orchestrates a complete machine learning pipeline for fish weight prediction including:
1. Fish dataset loading and validation
2. Feature engineering (species encoding, scaling)
3. Model training with hyperparameter tuning (Linear, Ridge, Lasso, Random Forest)
4. Model evaluation and performance analysis
5. Model deployment (optional)

Based on fish_regression.py and fish_analysis.py implementations.

Author: Fish ML pipeline
Date: July 2025
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import joblib
import json

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'fish_weight_prediction_pipeline',
    default_args=default_args,
    description='Complete ML pipeline for fish weight prediction using Fish.csv dataset',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    tags=['machine-learning', 'fish-prediction', 'regression', 'supervised-learning'],
)

# Global variables for data sharing between tasks
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/tmp')
DATA_DIR = os.path.join(AIRFLOW_HOME, 'ml_data')
MODEL_DIR = os.path.join(AIRFLOW_HOME, 'ml_models')

def setup_directories(**context):
    """Create necessary directories for the pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Created directories: {DATA_DIR}, {MODEL_DIR}")
    return "Directories setup completed"

def load_and_validate_data(**context):
    """Load Fish.csv data and perform initial validation."""
    print("Loading and validating Fish dataset...")
    
    try:
        # Load Fish dataset
        df = pd.read_csv('Dataset/Fish.csv')
        
        # Separate features and target
        X = df.drop('Weight', axis=1)
        y = df['Weight']
        
        # Data validation checks
        assert df.shape[0] > 0, "Dataset is empty"
        assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
        assert y.isnull().sum() == 0, "Target variable contains missing values"
        assert 'Species' in X.columns, "Species column missing"
        assert 'Weight' in df.columns, "Weight column missing"
        
        # Save data for next tasks
        data_path = os.path.join(DATA_DIR, 'raw_fish_data.pkl')
        joblib.dump({'X': X, 'y': y, 'df': df}, data_path)
        
        # Log dataset information
        data_info = {
            'dataset_name': 'Fish Weight Prediction',
            'dataset_shape': df.shape,
            'features': list(X.columns),
            'target': 'Weight',
            'species_count': len(df['Species'].unique()),
            'species_list': list(df['Species'].unique()),
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            },
            'feature_stats': {
                col: {
                    'mean': float(X[col].mean()) if X[col].dtype in ['int64', 'float64'] else 'categorical',
                    'std': float(X[col].std()) if X[col].dtype in ['int64', 'float64'] else 'categorical'
                } for col in X.columns
            },
            'validation_passed': True
        }
        
        info_path = os.path.join(DATA_DIR, 'fish_data_info.json')
        with open(info_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        print(f"   Fish dataset loaded and validated successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {list(X.columns)}")
        print(f"   Species: {list(df['Species'].unique())}")
        print(f"   Weight range: {y.min():.1f} - {y.max():.1f}g")
        
        return data_path
        
    except FileNotFoundError:
        print("❌ Fish.csv not found in Dataset/ directory")
        raise
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        raise

def preprocess_data(**context):
    """Preprocess fish data and split into train/test sets."""
    print(" Preprocessing fish data...")
    
    # Load data from previous task
    data_path = context['task_instance'].xcom_pull(task_ids='load_data')
    data = joblib.load(data_path)
    X, y = data['X'], data['y']
    
    # Handle categorical features (Species) - use label encoding for fish species
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_processed = X.copy()
    X_processed['Species'] = le.fit_transform(X['Species'])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Save preprocessed data and label encoder
    preprocessed_path = os.path.join(DATA_DIR, 'preprocessed_fish_data.pkl')
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': le,
        'species_mapping': dict(zip(le.transform(le.classes_), le.classes_))
    }, preprocessed_path)
    
    print(f" Fish data preprocessing completed")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]} (including encoded Species)")
    print(f"   Species encoded: {list(le.classes_)}")
    
    return preprocessed_path

def feature_engineering(**context):
    """Perform feature engineering specifically for fish weight prediction."""
    print(" Performing fish-specific feature engineering...")
    
    # Load preprocessed data
    data_path = context['task_instance'].xcom_pull(task_ids='preprocess_data')
    data = joblib.load(data_path)
    
    # Create feature engineering pipeline for fish data
    # Fish weight prediction benefits from:
    # 1. Standard scaling for physical measurements
    # 2. Polynomial features for non-linear relationships
    # 3. Feature selection for most relevant measurements
    
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # Degree 2 polynomial features work well for fish measurements
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        # Select top features based on correlation with weight
        ('feature_selection', SelectKBest(score_func=f_regression, k=15))
    ])
    
    # Fit feature engineering pipeline on training data
    X_train_engineered = feature_pipeline.fit_transform(data['X_train'], data['y_train'])
    X_test_engineered = feature_pipeline.transform(data['X_test'])
    
    # Save engineered features and pipeline
    engineered_path = os.path.join(DATA_DIR, 'engineered_fish_data.pkl')
    joblib.dump({
        'X_train': X_train_engineered,
        'X_test': X_test_engineered,
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'feature_pipeline': feature_pipeline,
        'label_encoder': data['label_encoder'],
        'species_mapping': data['species_mapping']
    }, engineered_path)
    
    print(f"   Fish feature engineering completed")
    print(f"   Original features: {data['X_train'].shape[1]}")
    print(f"   Engineered features: {X_train_engineered.shape[1]}")
    print(f"   Feature engineering: scaling + polynomial + selection")
    
    return engineered_path

def train_models(**context):
    """Train multiple models for fish weight prediction with hyperparameter tuning."""
    print(" Training fish weight prediction models...")
    
    # Load engineered data
    data_path = context['task_instance'].xcom_pull(task_ids='feature_engineering')
    data = joblib.load(data_path)
    
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Define models and their hyperparameters optimized for fish weight prediction
    models_config = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {'fit_intercept': [True, False]}
        },
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
        },
        'Lasso': {
            'model': Lasso(),
            'params': {'alpha': [0.1, 1.0, 10.0, 50.0]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    best_model = None
    best_score = float('-inf')
    best_model_name = None
    results = {}
    
    # Train and tune each model
    for model_name, config in models_config.items():
        print(f"   Training {model_name} for fish weight prediction...")
        
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = model_name
        
        results[model_name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        
        print(f"     Best CV score: {grid_search.best_score_:.4f}")
    
    # Save best model and results
    model_path = os.path.join(MODEL_DIR, 'best_fish_model.pkl')
    joblib.dump(best_model, model_path)
    
    results_path = os.path.join(MODEL_DIR, 'fish_training_results.json')
    results['best_model'] = best_model_name
    results['best_cv_score'] = best_score
    results['task'] = 'fish_weight_prediction'
    results['dataset'] = 'Fish.csv'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"   Fish weight prediction model training completed")
    print(f"   Best model: {best_model_name}")
    print(f"   Best CV score: {best_score:.4f}")
    
    return model_path

def evaluate_model(**context):
    """Evaluate the best fish weight prediction model on test data."""
    print(" Evaluating fish weight prediction model...")
    
    # Load model and test data
    model_path = context['task_instance'].xcom_pull(task_ids='train_models')
    data_path = context['task_instance'].xcom_pull(task_ids='feature_engineering')
    
    best_model = joblib.load(model_path)
    data = joblib.load(data_path)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage error for fish weight prediction
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Save evaluation results
    evaluation = {
        'model_type': 'fish_weight_prediction',
        'dataset': 'Fish.csv',
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
        'test_mape': mape,
        'evaluation_date': datetime.now().isoformat(),
        'test_samples': len(y_test),
        'weight_range': {
            'min': float(y_test.min()),
            'max': float(y_test.max()),
            'mean': float(y_test.mean())
        }
    }
    
    eval_path = os.path.join(MODEL_DIR, 'fish_evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    print(f"   Fish weight prediction model evaluation completed")
    print(f"   Test RMSE: {rmse:.2f}g")
    print(f"   Test R²: {r2:.4f}")
    print(f"   Test MAE: {mae:.2f}g")
    print(f"   Test MAPE: {mape:.2f}%")
    
    # Set success criteria for fish weight prediction
    if r2 > 0.85:  # Higher threshold for fish data as it's typically more predictable
        print(" Excellent fish weight prediction performance!")
        return eval_path
    elif r2 > 0.7:
        print(" Good fish weight prediction performance!")
        return eval_path
    else:
        print("⚠️  Fish weight prediction performance below threshold. Consider retraining.")
        return eval_path

def generate_model_report(**context):
    """Generate a comprehensive fish weight prediction model report."""
    print("🔄 Generating fish weight prediction model report...")
    
    # Load all results
    training_results_path = os.path.join(MODEL_DIR, 'fish_training_results.json')
    evaluation_results_path = context['task_instance'].xcom_pull(task_ids='evaluate_model')
    data_info_path = os.path.join(DATA_DIR, 'fish_data_info.json')
    
    with open(training_results_path, 'r') as f:
        training_results = json.load(f)
    
    with open(evaluation_results_path, 'r') as f:
        evaluation_results = json.load(f)
    
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    
    # Create comprehensive report
    report = {
        'pipeline_run_date': datetime.now().isoformat(),
        'project_name': 'Fish Weight Prediction ML Pipeline',
        'dataset_info': data_info,
        'model_training': training_results,
        'model_evaluation': evaluation_results,
        'model_ready_for_deployment': evaluation_results['test_r2'] > 0.7,
        'business_metrics': {
            'prediction_accuracy': f"{evaluation_results['test_r2']:.1%}",
            'average_error': f"{evaluation_results['test_mae']:.1f}g",
            'error_percentage': f"{evaluation_results['test_mape']:.1f}%"
        },
        'recommendations': {
            'deployment_ready': evaluation_results['test_r2'] > 0.85,
            'performance_level': 'excellent' if evaluation_results['test_r2'] > 0.9 else 
                               'good' if evaluation_results['test_r2'] > 0.8 else
                               'acceptable' if evaluation_results['test_r2'] > 0.7 else 'needs_improvement'
        }
    }
    
    report_path = os.path.join(MODEL_DIR, 'fish_ml_pipeline_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Fish weight prediction model report generated: {report_path}")
    print(f"   Model performance: {report['recommendations']['performance_level']}")
    print(f"   Prediction accuracy: {report['business_metrics']['prediction_accuracy']}")
    print(f"   Average error: {report['business_metrics']['average_error']}")
    
    return report_path

def deploy_model(**context):
    """Deploy fish weight prediction model (placeholder for actual deployment)."""
    print(" Deploying fish weight prediction model...")
    
    evaluation_results_path = context['task_instance'].xcom_pull(task_ids='evaluate_model')
    with open(evaluation_results_path, 'r') as f:
        evaluation = json.load(f)
    
    if evaluation['test_r2'] > 0.7:  # Adjusted threshold for fish prediction
        # In a real scenario, this would deploy to a server, create an API, etc.
        deployment_info = {
            'deployment_date': datetime.now().isoformat(),
            'model_version': f"fish_v_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'deployment_status': 'success',
            'model_performance': evaluation['test_r2'],
            'model_type': 'fish_weight_prediction',
            'dataset': 'Fish.csv',
            'deployment_target': 'fish_weight_api_endpoint',
            'model_metrics': {
                'r2_score': evaluation['test_r2'],
                'rmse': evaluation['test_rmse'],
                'mae': evaluation['test_mae'],
                'mape': evaluation['test_mape']
            }
        }
        
        deployment_path = os.path.join(MODEL_DIR, 'fish_deployment_info.json')
        with open(deployment_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"   Fish weight prediction model deployed successfully!")
        print(f"   Version: {deployment_info['model_version']}")
        print(f"   Performance: R² = {evaluation['test_r2']:.4f}")
        print(f"   RMSE: {evaluation['test_rmse']:.2f}g")
        print(f"   MAPE: {evaluation['test_mape']:.2f}%")
        
        return deployment_path
    else:
        print("❌ Fish model performance too low for deployment")
        print(f"   Current R²: {evaluation['test_r2']:.4f}, Required: > 0.7")
        raise ValueError("Fish weight prediction model performance below deployment threshold")

# Define tasks
setup_task = PythonOperator(
    task_id='setup_directories',
    python_callable=setup_directories,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_and_validate_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_model_report,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Set task dependencies
setup_task >> load_data_task >> preprocess_task >> feature_engineering_task
feature_engineering_task >> train_models_task >> evaluate_model_task
evaluate_model_task >> generate_report_task >> deploy_model_task

# Optional: Add email notification on completion
# email_notification = EmailOperator(
#     task_id='send_email',
#     to=['your-email@example.com'],
#     subject='ML Pipeline Completed',
#     html_content='<p>Your ML pipeline has completed successfully!</p>',
#     dag=dag,
# )
# deploy_model_task >> email_notification
