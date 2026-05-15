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

Locally deployed Ollama running open-source LLMs (e.g., Llama 3, Mistral) can be integrated into the Fish.csv regression pipeline as a **semantic reasoning layer** alongside scikit-learn and XGBoost. With the dataset fully known — 159 fish samples, 7 species (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt), physical measurements (Length1, Length2, Length3, Height, Width), and the regression target `Weight` (0–1650 g) — the LLM's role shifts from unstructured dataset discovery to domain-informed analysis: explaining *why* certain features predict weight, reasoning about feature correlation among the three length measurements, and producing human-readable interpretations of model predictions [1, 3].

Because Ollama runs entirely on the local machine, no fish measurement data is transmitted to external APIs. This also eliminates per-token costs and allows unrestricted experimentation with large prompt payloads (e.g., sending all 159 rows to the model context) [1, 3].

### How LLMs Assist with the Fish Dataset

The Fish.csv dataset is well-structured and the task is clearly defined: predict a fish's weight in grams from its species and five physical dimensions. In this known-dataset context, Ollama is useful for three distinct purposes:

1. **Feature reasoning**: Length1, Length2, and Length3 are three related but non-identical body length measurements. An LLM can articulate the biological relationships between them and Weight, helping decide which to retain or combine before statistical validation.
2. **Guided feature engineering**: Ask the LLM to propose derived features (e.g., a volume proxy `Length3 × Height × Width`) that may improve regression accuracy beyond the raw columns.
3. **Post-prediction explanation**: After training, feed SHAP importance scores back to the LLM to generate plain-language summaries of model behaviour, useful for communicating results to a non-technical audience.

### Feature Selection with Ollama

#### Contextual Reasoning for the Fish Features

The six input features (`Species`, `Length1`, `Length2`, `Length3`, `Height`, `Width`) have clear biological interpretations, and an LLM can reason about them with domain knowledge:

- **Semantic Ranking**: Ask the LLM to rank features by expected predictive power for fish weight. Biologically, body volume (approximated by length × height × width) drives weight, so the model is expected to rank `Length3`, `Height`, and `Width` highly.
- **Multicollinearity Reasoning**: Length1, Length2, and Length3 (vertical, diagonal, and cross lengths respectively) are strongly correlated. The LLM can explain *why* they overlap and suggest which single length measurement or linear combination is most informative.
- **Species as a Categorical Moderator**: The LLM can explain that `Species` acts as an interaction term — a Bream and a Smelt of the same length have very different weights — and recommend encoding strategies (one-hot vs. label encoding vs. target encoding).

#### Zero-Shot Feature Importance Scoring

Ask the LLM to score each feature before any model training:

1. Provide the feature list (`Species`, `Length1`, `Length2`, `Length3`, `Height`, `Width`) and the target (`Weight`).
2. Request an importance score (0–10) with biological justification for each feature.
3. Use the LLM's reasoning to form a prior expectation — then verify or challenge it with statistical methods below.

#### Statistical Validation of Selected Features

Validate LLM-suggested feature importance using the following methods on the Fish dataset:

- **Filter Methods (Fastest)**:
  - Pearson Correlation: Length1, Length2, and Length3 are expected to show high pairwise correlation (> 0.95). Identify and consider dropping the most redundant of the three.
  - Mutual Information (`mutual_info_regression`): Captures non-linear dependencies; `Species` may score higher here than under linear correlation because its weight effect is species-specific.
- **Wrapper Methods (Accurate)**:
  - Recursive Feature Elimination (RFE) with LinearRegression or RandomForest to iteratively remove the least important feature among the three length columns.
- **Embedded Methods (Robust)**:
  - Lasso (L1 Regularization) naturally zeros out redundant length coefficients, leaving only the most predictive length measurement alongside `Height`, `Width`, and `Species`.
- **Statistical Significance**:
  - F-Regression / Mutual Information: Use `f_regression` for linear relationships or `mutual_info_regression` for non-linear dependencies.
  - P-values: Features with $p > 0.05$ add noise rather than signal and should be candidates for removal.

### Regression Workflow with Ollama

#### Model Selection Guidance for Fish Weight Prediction

Given the Fish dataset characteristics (159 samples, 6 features, one categorical column, right-skewed weight distribution), an LLM can reason about algorithm trade-offs:

- **Linear Regression**: Fast and interpretable, but assumes linearity. Fish weight scales roughly with body volume (a cubic relationship), so raw length features violate linearity assumptions. The LLM can suggest log-transforming `Weight` or adding polynomial features.
- **Random Forest / Gradient Boosting**: Handles the non-linear weight-volume relationship and the categorical `Species` column naturally. The LLM can recommend starting hyperparameter ranges (`n_estimators=100–500`, `max_depth=4–8`) based on the dataset size.
- **Ridge / Lasso Regression**: Appropriate when multicollinearity among the three length features inflates coefficient variance. The LLM can explain the regularization trade-off and recommend a cross-validated alpha search range.

#### LLM-Assisted Feature Engineering

Beyond selecting from existing columns, the LLM can propose new derived features specific to fish morphology:

- **Volume Proxy**: `Length3 × Height × Width` approximates body volume and may capture the cubic weight-length relationship more directly than individual length columns.
- **Aspect Ratios**: `Height / Width` or `Length1 / Length3` may differentiate species that share similar lengths but differ in body shape.
- **Log Transforms**: `log(Weight)` linearises the allometric growth relationship (weight ∝ length³), improving the fit of linear models.

#### Post-Prediction Interpretation

After training, Ollama converts numeric SHAP importance scores into domain-grounded explanations for the Fish dataset:

- Feed SHAP values per feature to the LLM and ask for a biological interpretation of the feature rankings.
- Ask the LLM to explain individual predictions: *"A Perch with Length3 = 28 cm, Height = 8.5 cm, Width = 4.2 cm was predicted to weigh 320 g. Why?"*
- **Residual Analysis**: Plot residuals vs. predicted values — they should be randomly distributed around zero. If the LLM is given the residual plot description, it can suggest transformations (e.g., log-scale target) to correct systematic bias in the low-weight or high-weight range.

### End-to-End Workflow for Fish Weight Prediction

$$\text{Fish.csv} \rightarrow \text{Ollama Feature Reasoning} \rightarrow \text{Statistical Validation} \rightarrow \text{Regression Model} \rightarrow \text{Ollama Interpretation}$$

| Stage | Tool / Method | Fish Dataset Context |
|---|---|---|
| Data Loading & EDA | pandas, matplotlib | 159 samples, 7 species, Weight 0–1650 g |
| LLM Feature Reasoning | Ollama (Llama 3 / Mistral) | Rank Length1/2/3, Height, Width, Species for predicting Weight |
| Statistical Validation | f_regression, mutual_info, RFE, Lasso | Identify redundancy among correlated length features |
| Model Training | LinearRegression, RandomForest, XGBoost | Predict Weight; compare RMSE and R² across models |
| Interpretation | Ollama + SHAP | Explain why Species and body volume dominate predictions |
| Validation | Cross-validation (k=5) | Confirm feature set generalises across all 7 species |

### Practical Implementation

#### Step 1: LLM Feature Reasoning on the Known Dataset

```python
import ollama
import pandas as pd

df = pd.read_csv('Dataset/Fish.csv')
column_info = df.dtypes.to_string()
sample_data = df.head(5).to_string()

# Dataset is known: target is Weight, features are Species + 5 physical measurements.
# Use the LLM to reason about feature relationships, not to discover the target.
prompt = f"""
You are a data scientist working on a fish weight prediction task.
The dataset is Fish.csv with 159 samples and the following columns:

Column types:
{column_info}

Sample data (first 5 rows):
{sample_data}

The regression target is 'Weight' (grams). The features are:
- Species: categorical (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt)
- Length1: vertical length (cm)
- Length2: diagonal length (cm)
- Length3: cross length (cm)
- Height: body height (cm)
- Width: body width (cm)

Tasks:
1. Rank the five numeric features by expected predictive importance for Weight and justify each ranking with biological reasoning.
2. Explain the likely multicollinearity between Length1, Length2, and Length3, and recommend which to prioritise or combine.
3. Describe how Species should be encoded for regression (label encoding vs. one-hot) and why.
4. Suggest one or two derived features (e.g., a volume proxy) that may improve model accuracy.
"""

response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])
```

#### Step 2: Statistical Validation of Feature Importance

```python
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('Dataset/Fish.csv')

# Encode Species for statistical methods
df_encoded = df.copy()
le = LabelEncoder()
df_encoded['Species'] = le.fit_transform(df_encoded['Species'])

X = df_encoded.drop('Weight', axis=1)
y = df_encoded['Weight']

# F-regression: linear relationships with Weight
f_scores, p_values = f_regression(X, y)
print("F-scores:", dict(zip(X.columns, f_scores.round(2))))
print("P-values:", dict(zip(X.columns, p_values.round(4))))

# Mutual Information: captures non-linear and species-specific effects
mi_scores = mutual_info_regression(X, y, random_state=42)
print("Mutual Information:", dict(zip(X.columns, mi_scores.round(4))))

# Pearson Correlation matrix — check multicollinearity among Length columns
print("\nCorrelation with Weight:")
print(df_encoded.corr()['Weight'].round(3))

# RFE: identify which length measurement survives iterative elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

rfe = RFE(estimator=LinearRegression(), n_features_to_select=4)
rfe.fit(X, y)
print("\nRFE-selected features:", X.columns[rfe.support_].tolist())

# Lasso: zeros out the most redundant length column(s)
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=1.0)
lasso.fit(X_scaled, y)
coef_table = dict(zip(X.columns, lasso.coef_.round(3)))
print("\nLasso coefficients:", coef_table)
lasso_selected = [col for col, coef in coef_table.items() if coef != 0]
print("Lasso-retained features:", lasso_selected)
```

#### Step 3: LLM-Guided Feature Engineering

```python
import ollama
import pandas as pd

df = pd.read_csv('Dataset/Fish.csv')

# Add a volume proxy and ask the LLM whether it improves interpretability
df['Volume_proxy'] = df['Length3'] * df['Height'] * df['Width']
sample_engineered = df[['Species', 'Length3', 'Height', 'Width', 'Volume_proxy', 'Weight']].head(5).to_string()

prompt = f"""
A volume proxy feature (Length3 × Height × Width) was added to the Fish dataset.
Sample rows after feature engineering:

{sample_engineered}

1. Does this derived feature capture the allometric growth relationship (weight ∝ volume) better than the individual measurements?
2. Should the original Length3, Height, and Width columns be dropped after adding this proxy, or retained alongside it?
3. Would a log transformation of 'Weight' or 'Volume_proxy' further linearise the regression relationship?
Provide concise, actionable answers.
"""

response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])
```

#### Step 4: Post-Prediction Interpretation with Ollama

```python
import ollama

# After fitting your regression model and computing SHAP values on the Fish dataset
shap_summary = {
    'Species':  0.67,   # High: Bream vs. Smelt of same length differ greatly in weight
    'Length3':  0.52,   # Cross length — strongest individual length predictor
    'Height':   0.41,   # Body depth contributes to volume
    'Width':    0.35,   # Body width contributes to volume
    'Length1':  0.18,   # Vertical length — partially redundant with Length3
    'Length2':  0.14    # Diagonal length — most redundant of the three lengths
}

prompt = f"""
A Random Forest regression model was trained on the Fish.csv dataset to predict fish weight (grams).
The dataset contains 159 fish samples across 7 species: Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt.
Features: Species (categorical), Length1 (vertical), Length2 (diagonal), Length3 (cross), Height, Width.

The SHAP mean absolute feature importances are:
{shap_summary}

Tasks:
1. Explain in plain language why Species and Length3 are the top two predictors for fish weight.
2. Explain why Length1 and Length2 have lower importance than Length3 despite measuring the same fish.
3. Are there any importances that are surprising from a biological standpoint? Flag them and suggest a validation step.
4. Summarise the model behaviour in two sentences suitable for a non-technical audience.
"""

response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])
```

### Key Advantages for the Fish Dataset

- **Domain Reasoning**: Ollama provides biological context for feature relationships (e.g., allometric growth, species morphology) that purely statistical methods cannot supply [1, 3].
- **Multicollinearity Guidance**: With three correlated length features, the LLM helps decide which to keep or combine before running Lasso or RFE — reducing trial-and-error in feature engineering.
- **No API Costs**: All prompts run locally. Sending all 159 rows (or engineered features) to the model context incurs no per-token charges [1, 3].
- **Explainability**: Post-prediction SHAP summaries translated by the LLM into plain English make model outputs accessible to domain experts (e.g., fisheries biologists) who are unfamiliar with SHAP values.
- **Iterative Feedback**: After each training round, feed RMSE and feature importance back to the LLM to get targeted suggestions for the next iteration (e.g., try log(Weight) as the target, or add a volume proxy).

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
