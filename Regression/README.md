# Fish weight prediction using regression models

This project implements supervised machine learning models to predict fish weight using various physical measurements. The dataset contains information about 7 different fish species with their corresponding physical attributes.

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project)
3. [Setup](#setup)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Usage](#usage)
6. [Jupyter Notebook](#jupyter-notebook)
7. [Ollama Integration for Feature Selection and Regression](#ollama-integration-for-feature-selection-and-regression)
8. [Model Performance](#model-performance)
9. [Insights](#insights)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)
12. [License](#license)

## Dataset

The Fish.csv dataset is sourced from [Hugging Face](https://huggingface.co/datasets/scikit-learn/Fish/tree/main) and contains 161 samples with the following features:

### Features
- **Species**: Species name of fish (categorical variable with 7 species)
- **Weight**: Weight of fish in grams (target variable)
- **Length1**: Vertical length in cm
- **Length2**: Diagonal length in cm  
- **Length3**: Cross length in cm
- **Height**: Height in cm
- **Width**: Width in cm

## Project

```
Regression/
├── README.md
├── .gitignore
├── Dataset/
│   └── Fish.csv
├── requirements.txt
├── fish_regression.py          # Linear Regression example
├── fish_classification.py     # KNN Classification example
├── fish_analysis.py           # Complete analysis with multiple models
├── fish_analysis_notebook.ipynb  # Interactive Jupyter notebook
├── verify_dataset.py          # Dataset verification script
├── setup.sh                   # Automated setup script
└── quickstart.sh              # Quick start guide
```

## Setup

### 1. Create Python Virtual Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify activation (you should see (venv) in your prompt)
which python
```

#### Using conda (Alternative)
```bash
# Create conda environment
conda create -n fish_regression python=3.9

# Activate environment
conda activate fish_regression
```

### 2. Install required libraries

Install the required packages using pip:

```bash
# Make sure your virtual environment is activated
pip install --upgrade pip

# Install required packages
pip install scikit-learn pandas numpy matplotlib seaborn jupyter

# Or install from requirements.txt (after creating it)
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import sklearn, pandas, numpy, matplotlib; print('All libraries installed successfully!')"
```

### 4. Start Options

#### Option A: Script
```bash
./quickstart.sh
# Interactive menu to choose what to run
```

#### Option B: manual verification
```bash
python verify_dataset.py
# Verify dataset integrity before running ML models
```

## Machine Learning pipeline

### 1. Data loading and exploration
- Load the Fish.csv dataset using pandas
- Explore the dataset structure, missing values, and basic statistics
- Visualize data distributions and relationships

### 2. Data preprocessing
- Handle categorical variables (Species) using one-hot encoding
- Check for and handle missing values
- Scale numerical features if necessary
- Split data into training and testing sets

### 3. Model selection and training
- **Linear Regression**: For basic weight prediction
- **Polynomial Regression**: For capturing non-linear relationships
- **Random Forest**: For robust predictions with feature importance
- **KNN**: For classification tasks (species prediction)

### 4. Model evaluation
- Use appropriate metrics:
  - **Regression**: MSE, RMSE, R², MAE
  - **Classification**: Accuracy, Precision, Recall, F1-score
- Cross-validation for robust performance estimation
- Feature importance analysis

### 5. Prediction and deployment
- Make predictions on new data
- Save trained models for future use
- Create prediction functions for easy use

## Usage

### Basic Linear Regression for weight prediction

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data
df = pd.read_csv('Dataset/Fish.csv')

# Separate features and target
X = df.drop('Weight', axis=1)
y = df['Weight']

# Preprocessing
categorical_features = ['Species']
numerical_features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Apply preprocessing and split data
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.25, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.3f}")
```

### Species classification using KNN

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
fish = pd.read_csv('Dataset/Fish.csv')

# Features and target
X = fish.drop(['Species'], axis=1)
y = fish['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"KNN Accuracy: {accuracy:.3f}")

# Example prediction
sample = np.array([[340.0, 23.9, 26.5, 31.1, 12.3778, 4.6961]])
predicted_species = model.predict(sample)
print(f"Predicted species: {predicted_species[0]}")
```

## Running the examples

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Run the regression example:**
   ```bash
   python fish_regression.py
   ```

3. **Run the classification example:**
   ```bash
   python fish_classification.py
   ```

4. **Run complete analysis:**
   ```bash
   python fish_analysis.py
   ```

5. **🆕 Interactive Jupyter Notebook Analysis:**
   ```bash
   jupyter notebook fish_analysis_notebook.ipynb
   ```
   Or simply start Jupyter and open the notebook:
   ```bash
   jupyter notebook
   # Then navigate to fish_analysis_notebook.ipynb in the browser
   ```

## Jupyter Notebook

The **`fish_analysis_notebook.ipynb`** provides an interactive analysis environment with:

- **Interactive visualizations** with matplotlib and seaborn
- **Step-by-step data exploration** with detailed explanations
- **Multiple ML models** trained and compared side-by-side
- **Feature importance analysis** with visual plots
- **Interactive prediction function** to test new fish measurements
- **Performance metrics** and model comparisons
- **Species-wise analysis** and insights

### Notebook sections:
1. **Import libraries** - All necessary packages and configurations
2. **Data loading & exploration** - Dataset overview and basic statistics
3. **Data preprocessing** - Feature engineering and data preparation
4. **Exploratory data analysis** - Visualizations
5. **Model training** - Multiple regression models with pipelines
6. **Performance Comparison** - Visual model comparison and metrics
7. **Features** - Analysis of most predictive features
8. **Predictions** - Real-time prediction testing

## Model performance

- **Linear regression**: Good baseline performance (R² ≈ 0.85-0.95)
- **Polynomial Regression**: Better performance with non-linear relationships
- **Random forest**: Robust performance with feature importance insights
- **KNN classification**: High accuracy for species prediction (≈ 90-95%)

### What's KNN algorithm?

The K-Nearest Neighbors (KNN) algorithm is a non-parametric, supervised machine learning algorithm used for both classification and regression tasks.

**Choose the Value of K**

K represents the number of nearest neighbors to consider when making a prediction.

The choice of K is a hyperparameter that significantly impacts performance. A smaller K can lead to overfitting (sensitive to noise), while a larger K can lead to underfitting (losing fine-grained patterns).

Odd values for K are often preferred in classification to avoid ties in majority voting.

Cross-validation can be used to find the optimal K for your specific dataset.

**Find K-Nearest Neighbors**

Sort the calculated distances in ascending order.

Select the top K data points from the sorted list, as these are the nearest neighbors.

**Prediction**

For Classification:

Determine the most frequent class among the K nearest neighbors. This class becomes the predicted label for the new data point.

For Regression:

Calculate the average (or weighted average) of the target values of the K nearest neighbors. This average becomes the predicted value for the new data point. 

## Insights

1. **Length measurements** are typically the most important features for weight prediction
2. **Species** significantly influences the weight-to-size relationship
3. **Polynomial features** can capture non-linear growth patterns in fish
4. **Cross-validation** is essential for reliable performance estimation

## Troubleshooting

### Issues

1. **ModuleNotFoundError**: Make sure your virtual environment is activated and packages are installed
2. **File not found**: Ensure you're running scripts from the project root directory
3. **Memory issues**: If dataset is large, consider using batch processing or feature selection

- Check that your virtual environment is activated: `which python` should point to your venv
- Verify package versions: `pip list`
- Ensure you're in the correct directory: `pwd` should show the project root

## Ollama Integration for Feature Selection and Regression

### Overview

In the context of machine learning, locally deployed Ollama serves as a secure, private interface for open-source Large Language Models (LLMs) to perform semantic data analysis on "unknown" datasets, particularly during the early stages of a machine learning pipeline. For tasks like Feature Selection and Regression, Ollama allows these models to act as "semantic reviewers" that interpret feature names and data patterns without transmitting sensitive information to the cloud [1, 3].

Locally deployed Ollama running open-source LLMs (e.g., Llama 3, Mistral) is utilized to securely process "unknown" (unstructured/raw) datasets by acting as an intelligent intermediary to derive meaningful features (Feature Selection) and generate context-aware predictions (Regression) without sending sensitive data to external APIs.

### Processing Unknown Datasets

Ollama ensures that "unknown" or sensitive raw data stays on the local machine, which is critical for datasets in healthcare or finance that cannot be uploaded to public APIs:

- **Dataset Exploration**: Query large local files (e.g., 10 GB+ CSV or Parquet) to automatically summarize schema and data distributions before manual feature engineering begins.
- **Structured Feature Extraction**: For unstructured "unknown" data (such as text logs or financial news), Ollama helps extract structured features (e.g., sentiment scores or binary flags) into a tabular format suitable for regression training.

### Feature Selection with Ollama

When dealing with unknown datasets, Ollama helps interpret semantic meaning in columns or text fields that automated scripts might miss, mapping them to actionable numerical variables [2].

#### Contextual Feature Selection

- **Semantic Understanding**: LLMs analyze column names and metadata to identify features with high predictive power, even if data is poorly documented [2].
- **Feature Generation**: LLMs suggest, create, or transform features (feature engineering) based on domain-specific knowledge inferred from data labels [2].
- **Dimensionality Reduction**: LLMs act as an intelligent filter to remove irrelevant or redundant features based on reasoning rather than purely statistical correlation.

#### Zero-Shot and Iterative Prompting

For an unknown dataset, the first step is to extract its "meta-context" so the LLM can understand feature relationships:

1. Ask the LLM to identify which column is the most logical "target" for regression based on field names and sample values.
2. LLMs can perform feature selection by ranking variables based on their real-world semantic relevance to the target variable.
3. **Zero-Shot Scoring**: Ask the LLM to assign an "importance score" (0–10) to each feature.
4. **Filtering**: Remove columns that the LLM identifies as irrelevant or redundant (e.g., ID numbers or highly correlated features).

#### Statistical Validation of Selected Features

Validate the importance of selected features using statistical significance tests rather than just model scores:

- **Filter Methods (Fastest)**:
  - Use Pearson Correlation to identify and drop redundant features that are highly correlated with each other.
  - Use Mutual Information (`mutual_info_regression`) to capture non-linear dependencies between features and the target variable.
- **Wrapper Methods (Accurate)**:
  - Use Recursive Feature Elimination (RFE) with a baseline model like Linear Regression or Random Forest to iteratively remove the least important features.
- **Embedded Methods (Robust)**:
  - Use Lasso (L1 Regularization). Lasso naturally performs feature selection by shrinking the coefficients of unimportant features to zero.
- **Statistical Significance**:
  - F-Regression / Mutual Information: Use `f_regression` to check for linear relationships or `mutual_info_regression` for non-linear dependencies.
  - P-values: Check the p-values of regression coefficients; features with high p-values (typically $> 0.05$) often add noise rather than signal.

### Regression Workflow with Ollama

#### Model Selection and Hyperparameter Guidance

- **Model Selection**: Based on the description of the "unknown" dataset (size, sparsity, data types), the LLM can recommend suitable regression algorithms (e.g., Random Forest vs. Linear Regression) [3].
- **Hyperparameter Optimization**: LLMs can suggest reasonable ranges for hyperparameter tuning based on documentation processed in its local context.

#### Hybrid Reasoning: LLMs + Numerical Models

LLMs act as a "reasoning layer" that combines numerical predictions (e.g., from XGBoost) with textual data to refine regression outputs (e.g., predicting continuous variables like "days to event" or risk scores):

- **Data Transformation**: Use the Ollama API to convert raw tabular data into descriptive natural language prompts.
- **Iterative Feedback Loop**: Use a framework like LLM-FE or LLM-Lasso, where the LLM proposes feature transformations and you provide feedback based on a local validation score (e.g., RMSE).
- **Model-Generated Code**: Prompt models like Llama 3 to generate code for feature attribution methods, such as Lasso regression (embedded method) or Recursive Feature Elimination (RFE), to mathematically validate the initial semantic selection.

#### Post-Prediction Interpretation

After a regression model identifies important predictors (e.g., via SHAP values), Ollama can generate human-readable explanations in real-time, validating why the model selected specific features for a prediction, enhancing trust and auditability:

- **Residual Analysis**: Plot residuals vs. predicted values — they should be randomly distributed around zero without patterns (e.g., no "funnel" shape).

### End-to-End Workflow

The complete workflow for using locally deployed Ollama with a numerical ML model is:

$$\text{Data Preprocessing} \rightarrow \text{Ollama for Feature Engineering} \rightarrow \text{Numerical Model (XGBoost / Random Forest)} \rightarrow \text{Ollama for Interpretation}$$

| Stage | Tool / Method | Purpose |
|---|---|---|
| Data Preprocessing | pandas, scikit-learn | Cleaning, encoding, splitting |
| Feature Engineering | Ollama (Llama 3 / Mistral) | Semantic feature ranking and generation |
| Model Training | XGBoost, Random Forest | Numerical regression and prediction |
| Interpretation | Ollama + SHAP | Explainability and auditability |
| Validation | Cross-validation, RFE, LASSO | Confirm selected features on unseen data |

### Practical Implementation

#### Step 1: Extract Meta-Context from an Unknown Dataset

```python
import ollama
import pandas as pd

df = pd.read_csv('Dataset/Fish.csv')
column_info = df.dtypes.to_string()
sample_data = df.head(5).to_string()

prompt = f"""
You are a data scientist. Analyze the following dataset schema and sample rows.
Identify the most likely regression target variable and rank the remaining features
by their expected predictive importance for that target.

Column types:
{column_info}

Sample data:
{sample_data}
"""

response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])
```

#### Step 2: LLM-Guided Feature Selection (Zero-Shot Scoring)

```python
import ollama

feature_names = df.columns.tolist()
target = 'Weight'
features = [f for f in feature_names if f != target]

prompt = f"""
Given these features for a fish weight prediction regression task:
Features: {features}
Target: {target}

Assign an importance score (0-10) to each feature and explain your reasoning.
Identify any features that are likely redundant or should be removed.
"""

response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])
```

#### Step 3: Statistical Validation with scikit-learn

```python
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
df_encoded = df.copy()
le = LabelEncoder()
df_encoded['Species'] = le.fit_transform(df_encoded['Species'])

X = df_encoded.drop('Weight', axis=1)
y = df_encoded['Weight']

# F-regression for linear relationships
f_scores, p_values = f_regression(X, y)
print("F-scores:", dict(zip(X.columns, f_scores.round(2))))
print("P-values:", dict(zip(X.columns, p_values.round(4))))

# Mutual information for non-linear dependencies
mi_scores = mutual_info_regression(X, y, random_state=42)
print("Mutual Information:", dict(zip(X.columns, mi_scores.round(4))))

# Recursive Feature Elimination (RFE) — Wrapper Method
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

rfe = RFE(estimator=LinearRegression(), n_features_to_select=4)
rfe.fit(X, y)
selected = X.columns[rfe.support_].tolist()
print("RFE-selected features:", selected)

# LASSO — Embedded Method
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)
lasso_selected = X.columns[lasso.coef_ != 0].tolist()
print("LASSO-selected features:", lasso_selected)
```

#### Step 4: Post-Prediction Interpretation with Ollama

```python
import ollama

# After fitting your regression model and computing SHAP values
shap_summary = {
    'Length1': 0.45,
    'Length2': 0.38,
    'Length3': 0.52,
    'Height': 0.21,
    'Width': 0.18,
    'Species': 0.67
}

prompt = f"""
A regression model predicted fish weight. The SHAP feature importances are:
{shap_summary}

Provide a human-readable explanation of why these features matter for predicting
fish weight, and flag any that may be surprising or require validation.
"""

response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])
```

### Key Advantages

- **Data Privacy**: Ollama ensures all processing stays on the local machine. Sensitive or unknown datasets in healthcare or finance cannot be uploaded to public APIs [1, 3].
- **No API Costs**: Removes dependence on OpenAI/Anthropic APIs, reducing operational costs and latency for large datasets [1, 3].
- **Context-Aware Reasoning**: Combines the semantic understanding of LLMs with the mathematical precision of scikit-learn, XGBoost, and SHAP.
- **Auditability**: Post-prediction explanations generated by the LLM enhance trust in model outputs, which is critical for regulated industries.
- **Iterative Improvement**: Cross-validation ensures that LLM-selected and statistically validated features perform well on unseen data.

## References

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/index.html)
- [Fish Dataset on Hugging Face](https://huggingface.co/datasets/scikit-learn/Fish)
- [1] [Introduction to Supervised Machine Learning — Microsoft Premier Developer Blog](https://devblogs.microsoft.com/premier-developer/introduction-to-supervised-machine-learning/)
- [2] [scikit-learn: Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [3] [Ollama — Run Large Language Models Locally](https://ollama.com/)

## License

The dataset is available under the terms specified by Hugging Face.

---

**Note**: Always activate your virtual environment before running any Python scripts or installing packages.
