# Apache Airflow Fish weight prediction pipeline


### 1. **Fish weight prediction pipeline** (`fish_supervised_pipeline.py`)
- **Fish data loading**: Fish.csv dataset with species and physical measurements analysis
- **Species handling**: Automatic encoding of fish species (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt)
- **Feature engineering**: StandardScaler + PolynomialFeatures + SelectKBest optimized for fish measurements
- **Model training**: Multiple algorithms (Linear, Ridge, Lasso, RandomForest) optimized for fish weight prediction
- **Hyperparameter tuning**: GridSearchCV with 5-fold cross-validation optimized for fish data
- **Fish specific evaluation**: Complete metrics (MSE, RMSE, MAE, R², MAPE) with fish species analysis
- **Model persistence**: Save/load trained fish weight prediction models with species mapping

### 2. **Apache Airflow Fish pipeline orchestration** (`dags/ml_pipeline_dag.py`)
- **8-Stage Fish Weight Prediction Pipeline**:
  1. Setup directories for Fish data
  2. Fish dataset loading & validation (Fish.csv)
  3. Fish data preprocessing with species encoding
  4. Feature engineering optimized for Fish measurements
  5. Fish weight model training with hyperparameter tuning
  6. Fish weight model evaluation with species analysis
  7. Fish analysis Report Generation
  8. Fish weight prediction model deployment
- **Error fandling**: Retry logic and validation for fish data processing
- **Task dependencies**: Workflow orchestration for fish weight prediction
- **Monitoring**: Web UI with real-time status for fish pipeline

### 3. **Docker deployment for Fish weight prediction** (`Dockerfile`, `docker-compose.yml`)
- **Multi container**:
  - PostgreSQL database for Airflow metadata
  - Redis for fish pipeline task queue management
  - Airflow webserver with fish weight prediction web UI
  - Airflow scheduler for fish pipeline task execution
- **Production**: Proper volumes, networking, and health checks for fish ML workflows
- **Deployment**: Single `docker-compose up -d` command for fish weight prediction system

### 4. **Fish analysis tools and scripts**
- **`fish_regression.py`**: Original fish weight prediction using Linear Regression
- **`fish_analysis.py`**: Comprehensive fish analysis with multiple models
- **`fish_classification.py`**: Fish species classification analysis
- **`monitor_pipeline.py`**: Fish pipeline monitoring and health checks
- **`quickstart.sh`**: Interactive menu for fish analysis options
- **Configuration files**: Optimized for fish weight prediction workflows

### 5. **Fish documentation**
- **`README.md`**: Complete fish weight prediction project documentation
- **`AIRFLOW.md`**: Detailed deployment guide for fish prediction systems
- **`DOCKER.md`**: Docker deployment strategy and business value analysis
- **Code Documentation**: Detailed docstrings and comments for fish analysis

## 🐟 Fish dataset analysis

```
┌─────────────────────────────────────────────────────────────┐
│              Apache Airflow Fish Weight Prediction DAG      │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │Fish Data│ -> │Species  │ -> │ Model   │ -> │ Fish    │  │
│  │Loading  │    │Encoding │    │Training │    │Evaluation│  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │       │
│       v              v              v              v       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │Validate │    │Transform│    │Tune     │    │Deploy   │  │
│  │Fish.csv │    │Features │    │Fish     │    │Fish     │  │
│  │         │    │         │    │Models   │    │Model    │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
└─────────────────────────────────────────────────────────────┘
              │                                │                
              v                                v                
┌─────────────────────┐              ┌─────────────────────┐
│  Fish Data Storage  │              │ Fish Model Storage  │
│  - Fish.csv dataset │              │  - Trained models   │
│  - Species data     │              │  - Species mapping  │
│  - Physical measure │              │  - Weight predictions│
│  - Processed fish   │              │  - Fish evaluation  │
└─────────────────────┘              └─────────────────────┘
```

## Steps (Reference)

### 1. Initial setup
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Run standalone pipeline
```bash
source venv/bin/activate
python supervised_regression_pipeline.py
```

### 3. Start Airflow
```bash
./quickstart.sh
# Choose option 4 for both webserver + scheduler
```

### 4. Access Web UI
- **URL**: http://localhost:8080
- **Credentials**: admin/admin

### 5. Docker deployment
```bash
docker-compose up -d
```

##  Features

### Machine Learning components
 **Data Pipeline**: Automated data loading and validation
 **Feature Engineering**: Scaling, polynomial features, selection
 **Model Training**: Multiple algorithms with hyperparameter tuning
 **Evaluation**: Comprehensive metrics and visualizations
 **Model Persistence**: Save/load with metadata

### Apache Airflow integration
 **DAG definition**: Complete workflow as Airflow DAG
 **Task dependencies**: Proper execution order
 **Error Handling**: Retries and failure management
 **Monitoring**: Web UI with real-time status
 **Scheduling**: Configurable pipeline execution

### DevOps and Deployment
 **Docker containerization**: Multi-service deployment
 **Environment**: Virtual environments and dependencies
 **Configuration**: Environment variables and configs
 **Monitoring Tools**: Pipeline health and performance monitoring
 **Documentation**: Comprehensive guides and documentation

## Fish weight prediction

1. **Supervised learning**:   with Fish.csv labeled data
2. **Regression problem**: Predicting continuous fish weight values
3. **Pattern recognition**:  Multiple algorithms find patterns in fish measurements
4. **Label prediction**:  Trained models predict fish weight on new measurements
5. **Pipeline tasks**: Requested components:
   - Fish data preprocessing
   - Fish feature engineering
   - Fish weight model training
   - Hyperparameter tuning for fish data
   - Fish weight model evaluation
   - Fish weight model deployment

## Fish weight prediction features

### Fish weight prediction algorithms
- **Linear Regression**: Basic fish weight-measurement relationship modeling
- **Ridge Regression**: L2 regularization for fish weight overfitting prevention
- **Lasso Regression**: L1 regularization with fish feature selection
- **Random Forest**: Ensemble method for complex fish weight patterns

### Fish feature engineering pipeline
- **StandardScaler**: Fish measurement normalization for algorithm stability
- **PolynomialFeatures**: Non-linear fish measurement combinations (degree 2)
- **SelectKBest**: Statistical fish feature selection using f_regression
- **Species Encoding**: Automatic handling of fish species categories

### Fish weight evaluation metrics
- **MSE**: Mean Squared Error for fish weight prediction
- **RMSE**: Root Mean Squared Error for fish weight accuracy
- **MAE**: Mean Absolute Error for fish weight prediction
- **R²**: Coefficient of determination for fish weight model fit
- **MAPE**: Mean Absolute Percentage Error for fish weight business metrics
- **Cross-validation**: 5-fold CV for fish weight model selection

## 🐟 Fish dataset

### Fish weight prediction
- **Real-world Application**: Practical fish weight estimation
- **Species Diversity**: 7 different fish species for robust modeling
- **Physical Measurements**: Multiple length and dimension features
- **Business Relevance**: Applicable to fisheries and aquaculture

### Fish ML analysis
- **Existing Code**: Leverages your `fish_regression.py` and `fish_analysis.py`
- **Enhanced Pipeline**: Adds Airflow orchestration to existing fish analysis
- **Species Classification**: Includes fish species classification capabilities

## Docker deployment stack for Fish pipeline

- **PostgreSQL**: Persistent database for fish pipeline metadata
- **Redis**: Message broker for distributed fish analysis tasks
- **Airflow Webserver**: Web interface for fish pipeline on port 8080
- **Airflow Scheduler**: Fish weight prediction task scheduling engine

**Access the fish weight prediction pipeline at: http://localhost:8080**

---



