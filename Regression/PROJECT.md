# 🐟 Fish Machine Learning - Setup

### 📁 Project
```
Regression/
├── 📖 README.md                    # Comprehensive documentation
├── 🚫 .gitignore                   # Git ignore file (excludes venv, __pycache__, etc.)
├──  requirements.txt             # Python package dependencies
├── Dataset/
│   └── Fish.csv                    # Fish dataset (159 samples, 7 species)
├──  fish_regression.py           # Linear regression for weight prediction
├── 🐟 fish_classification.py       # KNN classification for species prediction  
├── fish_analysis.py             # Comprehensive analysis with multiple models
├──  verify_dataset.py            # Dataset verification (no ML libs required)
├──  quickstart.sh                # Interactive quick start script
└── 🛠️  setup.sh                    # Environment setup script
```

## Instructions

### Option 1: Interactive
```bash
./quickstart.sh
```

### Option 2: Manual setup
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install packages
pip install -r requirements.txt

# 3. Verify dataset
python3 verify_dataset.py

# 4. Run examples
python3 fish_regression.py      # Weight prediction
python3 fish_classification.py  # Species classification
python3 fish_analysis.py        # Full analysis
```

## Dataset

- **Total Samples**: 159 fish
- **Species**: 7 types (Perch, Bream, Roach, Pike, Smelt, Parkki, Whitefish)
- **Features**: Weight, Length1, Length2, Length3, Height, Width
- **Use Cases**: Weight prediction (regression) and species classification

## Machine Learning

### 1. Regression analysis (`fish_regression.py`)
- **Goal**: Predict fish weight from physical measurements
- **Models**: Linear Regression with one-hot encoding for species
- **Features**: 
  - Data preprocessing and exploration
  - Model training and evaluation
  - Feature importance analysis
  - Sample predictions
  - Prediction function creation

### 2. Classification analysis (`fish_classification.py`)
- **Goal**: Classify fish species from physical measurements
- **Models**: K-Nearest Neighbors with hyperparameter tuning
- **Features**:
  - Optimal k-value finding
  - Cross-validation
  - Confusion matrix analysis
  - Hyperparameter tuning with GridSearchCV
  - Confidence scoring

### 3. Analysis (`fish_analysis.py`)
- **Goal**: Complete analysis with multiple models
- **Regression Models**: Linear, Ridge, Lasso, Random Forest, Decision Tree, SVR
- **Classification Models**: KNN, Random Forest, Decision Tree, SVM, Naive Bayes
- **Features**:
  - Extensive data visualization
  - Model comparison
  - Feature importance analysis
  - Performance metrics
  - Key insights generation

## Technologies

- **Python 3.x**: Programming language
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **jupyter**: Interactive notebooks (optional)

##  Model performance

- **Regression (Weight Prediction)**: R² ≈ 0.85-0.95
- **Classification (Species Prediction)**: Accuracy ≈ 90-95%

## Visualizations

- Species distribution charts
- Correlation heatmaps
- Feature distribution plots
- Model performance comparisons
- Confusion matrices
- Actual vs predicted scatter plots
- Feature importance charts
- Cross-validation results

## Objectives

1. **Data Preprocessing**: Handle categorical variables, scaling, train/test splits
2. **Regression**: Predict continuous values (fish weight)
3. **Classification**: Predict categories (fish species)
4. **Model Evaluation**: Use appropriate metrics (R², accuracy, confusion matrix)
5. **Feature Engineering**: One-hot encoding, feature importance
6. **Hyperparameter Tuning**: GridSearchCV, cross-validation
7. **Visualization**: Data exploration and results presentation

## Troubleshooting

### Issues:
1. **Import errors**: Make sure virtual environment is activated
2. **File not found**: Run scripts from project root directory
3. **Python command**: Use `python3` instead of `python` on Linux

### Solutions:
```bash
# Activate environment
source venv/bin/activate

# Check current directory
pwd  # Should end with /Regression

# Verify Python
which python3

# Test imports
python3 -c "import pandas, sklearn; print('OK')"
```

## Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Fish Dataset on Hugging Face](https://huggingface.co/datasets/scikit-learn/Fish)

---

Run `./quickstart.sh` to begin, or follow the manual setup instructions above.
