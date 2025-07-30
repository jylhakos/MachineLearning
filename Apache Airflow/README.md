# Apache Airflow Fish weight prediction pipeline

A supervised learning project with Apache Airflow orchestration for fish weight prediction using regression. This project demonstrates a machine learning pipeline including data preprocessing, feature engineering, model training, hyperparameter tuning, model evaluation, and optional Docker deployment specifically for Fish.csv dataset analysis.

## Project

This project implements a supervised learning regression pipeline for fish weight prediction that:

1. **Fish data loading & analysis**: Loads Fish.csv dataset with species and physical measurements
2. **Feature engineering**: Applies species encoding, scaling, polynomial features, and feature selection
3. **Model training**: Trains multiple regression algorithms optimized for fish weight prediction
4. **Model evaluation**: Evaluates models using fish-specific metrics and visualizations
5. **Orchestration**: Uses Apache Airflow to manage the entire fish ML pipeline
6. **Deployment**: Optional Docker containerization for fish weight prediction API

## 🐟 Fish Dataset features

### Input features
- **Species**: Fish species (categorical: Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt)
- **Length1**: Vertical length (cm)
- **Length2**: Diagonal length (cm) 
- **Length3**: Cross length (cm)
- **Height**: Height (cm)
- **Width**: Diagonal width (cm)

### Target variable
- **Weight**: Fish weight in grams (continuous target for regression)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Fish Data Loading│ -> │Species Encoding │ -> │Feature Engineer │
│   (Fish.csv)    │    │ & Preprocessing │    │   (Scaling)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Model Training  │ -> │ Hyperparameter  │ -> │ Fish Weight     │
│(Multiple Algos) │    │    Tuning       │    │  Evaluation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Fish Report     │ -> │ Model Deploy.   │ -> │ Weight Predict. │
│  Generation     │    │   (API Ready)   │    │   Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Fish weight prediction pipeline
- **Multiple Algorithms**: Linear Regression, Ridge, Lasso, Random Forest optimized for fish data
- **Species Handling**: Automatic encoding of fish species (Bream, Roach, Whitefish, etc.)
- **Feature Engineering**: StandardScaler, PolynomialFeatures, SelectKBest for fish measurements
- **Hyperparameter Tuning**: GridSearchCV with cross-validation optimized for fish weight prediction
- **Fish-Specific Evaluation**: MAPE, weight distribution analysis, species-based evaluation
- **Model Persistence**: Save/load trained fish weight prediction models

### Apache Airflow orchestration
- **DAG based workflow**: Complete fish weight prediction pipeline as an Airflow DAG
- **Task dependencies**: Proper task ordering and dependencies for fish data processing
- **Error handling**: Robust error handling and retries for fish dataset processing
- **Monitoring**: Web UI for fish prediction pipeline monitoring
- **Scheduling**: Configurable fish model retraining schedules

### Deployment options
- **Local**: Virtual environment setup for fish analysis
- **Docker**: Complete containerized fish weight prediction solution
- **Production**: PostgreSQL backend, Redis for task queue, optimized for fish ML workflows

## Start

### 1. Setup environment

```bash
# Clone and navigate to project directory
cd "/home/laptop/EXERCISES/MACHINE LEARNING/MachineLearning/Apache Airflow"

# Run setup script
chmod +x setup.sh
./setup.sh
```

### 2. Run Fish weight prediction pipeline

```bash
# Activate virtual environment
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow_home

# Run the fish weight prediction pipeline
python fish_supervised_pipeline.py

# Or run the original fish regression analysis
python fish_regression.py

# Or run comprehensive fish analysis
python fish_analysis.py
```

### 3. Start Airflow services

```bash
# Start webserver and scheduler
./quickstart.sh

# Or manually:
airflow webserver --port 8080 --daemon
airflow scheduler --daemon
```

### 4. Access Airflow Web UI

- **URL**: http://localhost:8080
- **Username**: admin
- **Password**: admin

## 🐳 Docker deployment

### Prerequisites
- Docker and Docker Compose installed

### Docker start

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### Services
- **PostgreSQL**: Database for Airflow metadata
- **Redis**: Message broker for Celery
- **Airflow Webserver**: Web interface (port 8080)
- **Airflow Scheduler**: Task scheduling and execution

## 📁 Project Structure

```
Apache Airflow/
├── dags/
│   └── ml_pipeline_dag.py          # Airflow DAG for fish weight prediction
├── Dataset/
│   └── Fish.csv                    # Fish dataset with species and measurements
├── fish_supervised_pipeline.py     # Enhanced fish weight prediction pipeline
├── fish_regression.py              # Original fish regression analysis
├── fish_classification.py          # Fish species classification
├── fish_analysis.py               # Comprehensive fish analysis
├── verify_dataset.py              # Fish dataset verification
├── monitor_pipeline.py            # Pipeline monitoring for fish models
├── requirements.txt               # Python dependencies with Airflow
├── Dockerfile                     # Docker configuration
├── docker-compose.yml            # Multi-container setup
├── setup.sh                      # Environment setup script
├── quickstart.sh                 # Quick start script for fish analysis
├── airflow.env                   # Airflow configuration
├── README.md                     # This file
├── AIRFLOW.md                    # Deployment guide
├── DOCKER.md                     # Docker deployment strategy
└── PROJECT.md            # Project summary
```

## Configuration

### Airflow configuration
- **Executor**: LocalExecutor (configurable to CeleryExecutor)
- **Database**: PostgreSQL (in Docker) or SQLite (local)
- **Scheduler**: Default Airflow scheduler
- **Web Server**: Port 8080

### Fish weight prediction configuration
- **Algorithms**: LinearRegression, Ridge, Lasso, RandomForest optimized for fish data
- **Cross-Validation**: 5-fold CV
- **Test Split**: 20% of data with species stratification
- **Feature Selection**: Top K features using f_regression
- **Species Encoding**: Label encoding for categorical species data

## Pipeline

### 1. Setup directories
Creates necessary directories for fish data and models.

### 2. Fish data loading & validation
- Loads Fish.csv dataset with species and physical measurements
- Validates data quality and completeness for fish weight prediction
- Saves fish data information and species statistics

### 3. Fish data preprocessing
- Splits data into training and testing sets with species stratification
- Encodes fish species using LabelEncoder
- Handles categorical and numerical features for fish measurements

### 4. Feature engineering for Fish weight prediction
- Applies StandardScaler for physical measurements (length, height, width)
- Creates polynomial features for non-linear relationships in fish measurements
- Selects top K features most relevant for fish weight prediction

### 5. Fish weight model training
- Trains multiple regression algorithms optimized for fish weight prediction
- Performs hyperparameter tuning with GridSearchCV for fish-specific parameters
- Selects best model based on cross-validation score for weight prediction

### 6. Fish weight model evaluation
- Evaluates best model on test fish data
- Calculates comprehensive metrics (MSE, RMSE, MAE, R², MAPE)
- Analyzes performance by fish species and weight ranges
- Checks for overfitting in fish weight prediction

### 7. Fish analysis report generation
- Creates comprehensive fish weight prediction report
- Includes fish dataset info, training results, and evaluation metrics
- Provides business insights for fish weight estimation accuracy
- Determines deployment readiness for fish weight prediction API

### 8. Fish model deployment
- Deploys model if fish weight prediction performance meets criteria
- Creates deployment metadata for fish weight prediction service
- (In production: would deploy to fish weight estimation API)

## Metrics and evaluation

### Fish weight prediction metrics
- **Mean Squared Error (MSE)**: Average squared differences in weight prediction
- **Root Mean Squared Error (RMSE)**: Square root of MSE in grams
- **Mean Absolute Error (MAE)**: Average absolute differences in grams
- **R-squared (R²)**: Coefficient of determination for weight variance explained
- **Mean Absolute Percentage Error (MAPE)**: Average percentage error in weight prediction

### Fish model selection Criteria
- **Cross-Validation Score**: Primary selection metric for fish weight prediction
- **Generalization**: Training vs. test performance for fish data
- **Deployment Threshold**: R² > 0.7 for fish weight prediction deployment
- **Species Analysis**: Performance consistency across different fish species

## Troubleshooting

### Issues

1. **Import errors**: Run `pip install -r requirements.txt`
2. **Airflow database issues**: Delete airflow_home and re-run setup.sh
3. **Port 8080 in use**: Change port in docker-compose.yml or stop conflicting service
4. **Permission denied**: Run `chmod +x setup.sh quickstart.sh`

### Commands

```bash
# Check Airflow processes
ps aux | grep airflow

# Stop Airflow services
pkill -f airflow

# View Airflow logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log

# Reset Airflow database
airflow db reset

# List DAGs
airflow dags list

# Test DAG
airflow dags test supervised_learning_pipeline $(date +%Y-%m-%d)
```

## Pipeline monitoring

### Airflow Web UI features
- **DAG View**: Visual pipeline representation
- **Task Status**: Real-time task execution status
- **Logs**: Detailed task execution logs
- **Metrics**: Pipeline performance metrics
- **Scheduling**: Configure pipeline schedules

### Performance monitoring
- Model performance metrics tracked in JSON files
- Pipeline execution times logged
- Error tracking and notification capabilities

## Extending the pipeline

### Adding algorithms
1. Update `models_config` in the training task
2. Add algorithm-specific hyperparameters
3. Update evaluation logic if needed

### Adding features
1. Modify feature engineering pipeline
2. Update feature selection parameters
3. Test with new feature combinations

### Custom Datasets
1. Update data loading function
2. Modify preprocessing steps as needed
3. Adjust evaluation metrics if necessary

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MLOps Best Practices](https://ml-ops.org/)
- [Python venv: How To Create, Activate, Deactivate, And Delete](https://python.land/virtual-environments/virtualenv)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
