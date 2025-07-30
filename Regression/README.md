# Fish weight prediction using regression models

This project implements supervised machine learning models to predict fish weight using various physical measurements. The dataset contains information about 7 different fish species with their corresponding physical attributes.

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
- **Comprehensive performance metrics** and model comparisons
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

## References

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/index.html)
- [Fish Dataset on Hugging Face](https://huggingface.co/datasets/scikit-learn/Fish)

## License

The dataset is available under the terms specified by Hugging Face.

---

**Note**: Always activate your virtual environment before running any Python scripts or installing packages.
