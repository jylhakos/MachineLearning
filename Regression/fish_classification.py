"""
Fish species classification using K-Nearest Neighbors
====================================================

This script demonstrates how to classify fish species using KNN algorithm
with the Fish.csv dataset. It includes data preprocessing, model training,
hyperparameter tuning, and example predictions.

Author: Fish ML
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the Fish dataset and perform basic exploration."""
    print("=" * 60)
    print("FISH SPECIES CLASSIFICATION USING K-NEAREST NEIGHBORS")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv('Dataset/Fish.csv')
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nSpecies distribution:")
    species_counts = df['Species'].value_counts()
    print(species_counts)
    
    print(f"\nNumber of unique species: {df['Species'].nunique()}")
    print(f"Species list: {list(df['Species'].unique())}")
    
    return df

def preprocess_data(df):
    """Preprocess the data for classification."""
    print("\n" + "=" * 40)
    print("DATA PREPROCESSING")
    print("=" * 40)
    
    # Features (X) and target (y)
    X = df.drop(['Species'], axis=1)  # All columns except Species
    y = df['Species']  # Species column
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Check for missing values
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")
    
    # Basic statistics
    print(f"\nFeature statistics:")
    print(X.describe())
    
    return X, y

def find_optimal_k(X, y):
    """Find the optimal value of k using cross-validation."""
    print("\n" + "=" * 40)
    print("FINDING OPTIMAL K VALUE")
    print("=" * 40)
    
    # Split data for k optimization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different k values
    k_range = range(1, min(31, len(X_train)))  # Up to 30 or training size
    scores = []
    
    print("Testing k values from 1 to", max(k_range))
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
        knn.fit(X_train_scaled, y_train)
        score = knn.score(X_test_scaled, y_test)
        scores.append(score)
        
        if k <= 10 or k % 5 == 0:  # Print selected k values
            print(f"k={k:2d}: Accuracy = {score:.4f}")
    
    # Find best k
    best_k = k_range[np.argmax(scores)]
    best_score = max(scores)
    
    print(f"\nBest k value: {best_k}")
    print(f"Best accuracy: {best_score:.4f}")
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs k Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return best_k, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test, best_k):
    """Train the KNN model with optimal k and evaluate performance."""
    print("\n" + "=" * 40)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 40)
    
    # Train the model with best k
    model = KNeighborsClassifier(
        n_neighbors=best_k, 
        metric='minkowski', 
        p=2,  # Euclidean distance
        weights='uniform'
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report (Test Set):")
    print("-" * 50)
    print(classification_report(y_test, y_test_pred))
    
    return model, y_test_pred

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    print("\n" + "=" * 40)
    print("CONFUSION MATRIX")
    print("=" * 40)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y_test.unique()), 
                yticklabels=sorted(y_test.unique()))
    plt.title('Confusion Matrix - Fish Species Classification')
    plt.xlabel('Predicted Species')
    plt.ylabel('Actual Species')
    plt.tight_layout()
    plt.show()
    
    # Print confusion matrix details
    species_list = sorted(y_test.unique())
    print("\nConfusion Matrix:")
    print("-" * 50)
    print("Rows = Actual, Columns = Predicted")
    print(f"{'Species':<12}", end='')
    for species in species_list:
        print(f"{species:<8}", end='')
    print()
    
    for i, actual_species in enumerate(species_list):
        print(f"{actual_species:<12}", end='')
        for j, _ in enumerate(species_list):
            print(f"{cm[i][j]:<8}", end='')
        print()

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\n" + "=" * 40)
    print("HYPERPARAMETER TUNING")
    print("=" * 40)
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # Only relevant for minkowski
    }
    
    print("Searching for best hyperparameters...")
    print("This may take a moment...")
    
    # Create KNN classifier
    knn = KNeighborsClassifier()
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def make_sample_predictions(model, scaler, df):
    """Make predictions on sample data."""
    print("\n" + "=" * 40)
    print("SAMPLE PREDICTIONS")
    print("=" * 40)
    
    # Get sample from each species
    species_samples = {}
    for species in df['Species'].unique():
        species_data = df[df['Species'] == species].iloc[0]
        species_samples[species] = species_data
    
    print("Predictions for sample fish from each species:")
    print("-" * 60)
    
    X_columns = df.drop(['Species'], axis=1).columns
    
    for species, sample in species_samples.items():
        # Prepare sample for prediction
        sample_features = sample[X_columns].values.reshape(1, -1)
        
        # Scale the features
        sample_scaled = scaler.transform(sample_features)
        
        # Make prediction
        predicted_species = model.predict(sample_scaled)[0]
        prediction_proba = model.predict_proba(sample_scaled)[0]
        confidence = max(prediction_proba)
        
        # Display results
        status = "✅" if predicted_species == species else "❌"
        print(f"{status} {species:<15}: Predicted={predicted_species:<15} "
              f"(Confidence: {confidence:.3f})")

def create_prediction_function(model, scaler, feature_columns):
    """Create a convenient prediction function."""
    def predict_fish_species(weight, length1, length2, length3, height, width):
        """
        Predict fish species based on physical measurements.
        
        Parameters:
        - weight: float, weight in grams
        - length1: float, vertical length in cm
        - length2: float, diagonal length in cm
        - length3: float, cross length in cm
        - height: float, height in cm
        - width: float, width in cm
        
        Returns:
        - predicted_species: str, predicted species name
        - confidence: float, prediction confidence (0-1)
        - probabilities: dict, probability for each species
        """
        # Create input array
        input_features = np.array([[weight, length1, length2, length3, height, width]])
        
        # Scale features
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        predicted_species = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities)
        
        # Create probability dictionary
        species_names = model.classes_
        prob_dict = dict(zip(species_names, probabilities))
        
        return predicted_species, confidence, prob_dict
    
    return predict_fish_species

def main():
    """Main function to run the complete analysis."""
    try:
        # Load and explore data
        df = load_and_explore_data()
        
        # Preprocess data
        X, y = preprocess_data(df)
        
        # Find optimal k and prepare data
        best_k, scaler, X_train_scaled, X_test_scaled, y_train, y_test = find_optimal_k(X, y)
        
        # Train and evaluate model
        model, y_pred = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, best_k)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred)
        
        # Hyperparameter tuning (optional)
        print("\nPerforming advanced hyperparameter tuning...")
        best_model = hyperparameter_tuning(X_train_scaled, y_train)
        
        # Evaluate best model
        best_y_pred = best_model.predict(X_test_scaled)
        best_accuracy = accuracy_score(y_test, best_y_pred)
        print(f"\nBest tuned model accuracy: {best_accuracy:.4f}")
        
        # Use the better model
        final_model = best_model if best_accuracy > accuracy_score(y_test, y_pred) else model
        
        # Sample predictions
        make_sample_predictions(final_model, scaler, df)
        
        # Create prediction function
        predict_species = create_prediction_function(final_model, scaler, X.columns)
        
        # Example predictions
        print("\n" + "=" * 40)
        print("EXAMPLE PREDICTIONS")
        print("=" * 40)
        
        # Example measurements
        examples = [
            (340.0, 23.9, 26.5, 31.1, 12.3778, 4.6961),  # Bream
            (714.0, 35.0, 38.5, 42.5, 15.2, 6.2),        # Pike  
            (40.0, 13.8, 15.0, 16.2, 4.8, 2.0)           # Roach
        ]
        
        print("\nExample predictions:")
        print("-" * 70)
        for weight, l1, l2, l3, h, w in examples:
            species, confidence, probabilities = predict_species(weight, l1, l2, l3, h, w)
            print(f"Input: W={weight}g, L1={l1}, L2={l2}, L3={l3}, H={h}, W={w}")
            print(f"  → Predicted: {species} (Confidence: {confidence:.3f})")
            
            # Show top 3 probabilities
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  → Top 3: {', '.join([f'{s}({p:.3f})' for s, p in sorted_probs])}")
            print()
        
        print("=" * 60)
        print("CLASSIFICATION ANALYSIS COMPLETE! 🐟")
        print("=" * 60)
        
        return final_model, scaler, predict_species
        
    except FileNotFoundError:
        print("Error: Dataset/Fish.csv not found!")
        print("Please make sure the Fish.csv file is in the Dataset/ folder.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

if __name__ == "__main__":
    model, scaler, predict_species = main()
