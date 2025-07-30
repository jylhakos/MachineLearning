# Machine Learning

Machine learning algorithms aim to learn and improve their accuracy as they process more datasets.

## Supervised Learning

Supervised learning uses algorithms to train a model to find patterns in a dataset with labels and features and then uses the trained model to predict the labels on a new dataset’s features.

A large amount of labeled training datasets are provided which provide examples of the data that the computer will be processing.

Supervised learning tasks can be categorized as classification or regression problems.

## Regression

Regression is a supervised learning technique used to predict continuous numerical values. Unlike classification which predicts discrete categories, regression models estimate relationships between input features and continuous target variables.

**Key Characteristics:**
- Predicts continuous numerical outputs
- Evaluates model performance using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R²
- Applications include price prediction, demand forecasting, and risk assessment

The `Regression/` folder contains implementations of various regression algorithms including:
- Linear Regression
- Polynomial Regression  
- Decision Tree Regression
- Fish weight prediction models using the Fish.csv dataset

### Linear Regression

Linear Regression models the relationship between features and target as a linear equation:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- ŷ is the predicted value
- β₀ is the y-intercept (bias term)
- β₁, β₂, ..., βₙ are coefficients (weights)
- x₁, x₂, ..., xₙ are input features
- ε is the error term

The algorithm finds optimal coefficients by minimizing the sum of squared residuals using methods like Ordinary Least Squares (OLS) or gradient descent. Linear regression assumes a linear relationship between features and target, making it interpretable but potentially limited for complex patterns.

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a non-parametric algorithm that makes predictions based on the k closest training examples in the feature space.

**Algorithm Steps:**
1. Calculate distance (typically Euclidean) between the query point and all training points
2. Select k nearest neighbors
3. For regression: Return the average of the k neighbors' values
4. For classification: Return the majority class among k neighbors

**Mathematical Foundation:**
For regression, the prediction is:
```
ŷ = (1/k) * Σ(i=1 to k) y_i
```

**Key Parameters:**
- k: Number of neighbors to consider
- Distance metric: Euclidean, Manhattan, Minkowski, etc.
- Weighting scheme: Uniform or distance-based weights

KNN is simple and effective but can be computationally expensive for large datasets and sensitive to the curse of dimensionality.

## Model Evaluation and Hyperparameters

Evaluating machine learning models requires understanding various metrics and optimization techniques to ensure robust performance and generalization to unseen data.

### Classification Metrics

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Measures the overall correctness of predictions, but can be misleading with imbalanced datasets.

**Precision:**
```
Precision = TP / (TP + FP)
```
Indicates the accuracy of positive predictions. High precision means fewer false positives.

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
Indicates the ability to identify all positive cases. High recall means fewer false negatives.

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall, balancing both metrics.

**AUC-ROC:**
Area under the ROC curve, which visualizes the trade-off between true positive rate and false positive rate. Values range from 0.5 (random) to 1.0 (perfect).

**Confusion Matrix:**
A table summarizing prediction outcomes, showing true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

### Regression Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/n) × Σ(yi - ŷi)²
```
Average of the squared differences between predicted and actual values.

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE
```
Square root of MSE, giving errors in the same units as the target variable.

**R-squared (R²):**
```
R² = 1 - (SS_res / SS_tot)
```
Indicates the proportion of variance in the dependent variable explained by the model. Values range from 0 to 1.

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ|yi - ŷi|
```
Average of absolute differences, less sensitive to outliers than MSE.

### Hyperparameters and Model Optimization

**Hyperparameters** are configuration settings that control the learning process and model behavior, distinct from parameters learned during training.

**Common Hyperparameters:**
- Learning rate (α): Controls step size in gradient descent
- Regularization strength (λ): Prevents overfitting
- Number of estimators: In ensemble methods
- Maximum depth: In tree-based models
- Batch size: For neural networks and gradient methods

**Hyperparameter Tuning Methods:**
1. **Grid Search:** Exhaustive search over parameter combinations
2. **Random Search:** Random sampling from parameter distributions
3. **Bayesian Optimization:** Uses probabilistic models to guide search
4. **Genetic Algorithms:** Evolutionary approach to optimization

### Cross-Validation and Model Selection

**Cross-Validation:**
A technique to evaluate how well a model generalizes to unseen data by partitioning the dataset into multiple folds.

**K-Fold Cross-Validation:**
```
CV_score = (1/k) × Σ(i=1 to k) score_i
```

**Common CV Strategies:**
- **K-Fold:** Dataset split into k equal parts
- **Stratified K-Fold:** Maintains class distribution in each fold
- **Time Series Split:** Respects temporal order for time-dependent data
- **Leave-One-Out (LOO):** Each sample serves as validation set

### Apache Airflow and Iterative Model Evaluation

Apache Airflow orchestrates automated model evaluation workflows that iterate through different hyperparameter combinations and track performance metrics:

**Automated Hyperparameter Optimization:**
- Airflow DAGs can execute grid search or random search tasks
- Each task evaluates different parameter combinations
- Results are logged and compared automatically

**Metric Tracking and Comparison:**
- Airflow integrates with MLflow or similar tools to track experiments
- Automated comparison of accuracy, precision, recall, F1-score across runs
- Performance trends monitored over time for model drift detection

**Pipeline Benefits:**
- **Reproducibility:** Consistent evaluation protocols across experiments
- **Scalability:** Parallel execution of hyperparameter combinations
- **Monitoring:** Real-time tracking of training progress and convergence
- **Automated Selection:** Programmatic selection of best-performing models

The `Apache Airflow/` implementation includes `monitor_pipeline.py` which demonstrates automated evaluation and model selection based on predefined performance thresholds and validation metrics.

## Ensemble learning

Ensemble learning algorithms combine multiple machine learning algorithms to obtain a better model.

## Decision trees

Decision tree learning is a machine learning approach that processes inputs using a series of classifications or regressions which lead to an answer or a continuous value.

Decision trees create a model that predicts the label by evaluating a tree of if-then-else true/false feature questions, and estimating the minimum number of questions needed to assess the probability of making a correct decision. 

Decision trees can be used for classification to predict a category, or regression to predict a continuous numeric value.

## Gradient Boosting

Gradient Boosting is an ensemble method that builds models sequentially, where each new model corrects errors made by previous models. The algorithm minimizes a loss function L(y, F(x)) by iteratively adding weak learners in the direction of the negative gradient.

**Mathematical Framework:**

The algorithm starts with an initial prediction F₀(x) and iteratively improves it:

```
F_m(x) = F_{m-1}(x) + γ_m * h_m(x)
```

Where:
- F_m(x) is the model after m iterations
- h_m(x) is the m-th weak learner
- γ_m is the step size (learning rate)

The weak learner h_m(x) is fitted to the negative gradient (pseudo-residuals):

```
r_{im} = -[∂L(y_i, F(x_i))/∂F(x_i)]_{F=F_{m-1}}
```

This approach effectively performs functional gradient descent in the space of functions, making it applicable to both classification and regression tasks. Early stopping mechanisms prevent overfitting by monitoring validation performance and halting training when improvement plateaus.

###  Classification

***Loss Function***

The loss function's purpose is to calculate how well the model predicts, given the available data.

***Weak Learner***

A weak learner classifies the data, but it makes a lot of mistakes.

***Additive Model***

The predictions are combined in an additive manner, where the addition of each base model improves (or boosts) the overall model. 

This is how the trees are added incrementally, iteratively, and sequentially. 

## XGBoost

Extreme Gradient Boosting (XGBoost) is implementation of gradient boosting.

With XGBoost, trees are built in parallel, instead of sequentially like Gradient Boosted Decision Trees (GBDT).

## Apache Airflow for Machine Learning Pipelines

Apache Airflow is an open-source platform designed to programmatically author, schedule, and monitor workflows. In machine learning contexts, it orchestrates complex data pipelines that handle the entire ML lifecycle from data ingestion to model deployment.

### Key Features for ML Workflows

**Directed Acyclic Graphs (DAGs):**
- Define workflow dependencies as Python code
- Ensure proper execution order of ML pipeline stages
- Enable parallel processing where dependencies allow

**Task Management:**
- Data extraction and preprocessing
- Feature engineering and transformation
- Model training and validation
- Model evaluation and comparison
- Model deployment and monitoring

**Scheduling and Monitoring:**
- Automated pipeline execution on time-based or event-based triggers
- Built-in retry mechanisms for failed tasks
- Comprehensive logging and alerting systems
- Web-based UI for monitoring pipeline status

### Machine Learning Pipeline Architecture

The `Apache Airflow/` folder contains a complete ML pipeline implementation:

1. **Data Pipeline Components:**
   - `fish_analysis.py`: Core data analysis and preprocessing
   - `fish_classification.py`: Classification model implementations
   - `fish_regression.py`: Regression model implementations
   - `verify_dataset.py`: Data quality validation

2. **API and Deployment:**
   - `fish_api.py`: REST API for model serving
   - `Dockerfile.fish-api`: Containerized API deployment
   - `deploy-fish-production.sh`: Production deployment scripts

3. **Pipeline Orchestration:**
   - `dags/ml_pipeline_dag.py`: Main Airflow DAG definition
   - `supervised_regression_pipeline.py`: End-to-end supervised learning pipeline
   - `monitor_pipeline.py`: Pipeline monitoring and alerting

### Workflow Benefits

- **Reproducibility:** Version-controlled pipeline definitions ensure consistent execution
- **Scalability:** Distributed execution across multiple workers
- **Fault Tolerance:** Automatic retries and error handling
- **Monitoring:** Real-time visibility into pipeline health and performance
- **Integration:** Seamless connection with databases, cloud services, and ML frameworks

The implementation demonstrates how Airflow transforms ad-hoc ML experiments into production-ready, automated workflows that can handle data drift, model retraining, and continuous deployment scenarios.

## Amazon SageMaker

Amazon SageMaker (AI) is a machine learning (ML) service designed to help developers build, train, and deploy ML models into production ready hosted environments.

### Properties

**Managed infrastructure**
- Eliminates the need to build and manage your own servers
- Provides scalable compute resources for training and inference
- Handles infrastructure provisioning and scaling automatically

**Integrated Development Environment**
- UI experience for running ML workflows
- Support for multiple IDEs and development environments
- Collaborative tools for data scientists and developers

**Data management**
- Store and share data without server management overhead
- Integration with Amazon S3 for data lakes and storage
- Built-in data processing and transformation capabilities

### Machine Learning workflow

**1. Data preparation and storage**
- Utilize Amazon S3 for scalable data storage
- Built-in data processing tools for cleaning and transformation
- Integration with other AWS data services

**2. Model development and training**
- **Built-in Algorithms:** Pre-built, optimized ML algorithms for common use cases
- **Bring-Your-Own-Algorithms:** Support for custom algorithms and frameworks
- **Distributed Training:** Flexible training options that scale with your data
- **Managed Training Jobs:** Automated infrastructure management during training

**3. Model deployment options**

**Real-time inference**
- **Amazon SageMaker Endpoints:** Deploy models for real-time predictions
- Auto-scaling capabilities based on traffic
- A/B testing support for model variants

**Batch processing**
- **Batch transform jobs:** Process large datasets in batches
- Cost-effective for bulk predictions
- Results stored directly in S3

**4. MLOps and pipeline**
- **Amazon SageMaker pipelines:** Automate end-to-end ML workflows
- **Model registry:** Version control and lifecycle management
- **Model monitoring:** Detect data drift and model performance degradation
- **Feature store:** Centralized repository for ML features

### Integration with Development Workflows

**Containerization**
- Native Docker container support
- Integration with Amazon Elastic Container Registry (ECR)
- Custom runtime environments for specific frameworks

**SDK and API access**
- Python SDK for programmatic access
- REST APIs for integration with existing systems
- CLI tools for automation and scripting

**Security**
- IAM integration for access control
- VPC support for network isolation
- Encryption at rest and in transit
- Compliance with industry standards

### ML for Development

- **Faster time-to-market:** Reduced infrastructure overhead enables focus on model development
- **Scalability:** Automatic scaling from experimentation to production workloads
- **Cost Optimization:** Pay-per-use pricing model with no upfront costs
- **Collaboration:** Shared environments and tools for team-based development
- **Production-Ready:** Built-in best practices for ML operations and deployment

Amazon SageMaker transforms the traditional ML development process by providing a comprehensive platform that handles the complexities of infrastructure management while maintaining flexibility for custom workflows and algorithms.

### References

[Classification: Accuracy, recall, precision, and related metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

[Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression)

[What is Amazon SageMaker AI?](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
