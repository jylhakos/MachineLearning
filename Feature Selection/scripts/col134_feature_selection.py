"""
col134_feature_selection.py
============================
Purpose
-------
Given a CSV dataset that contains a column named COL_134, this script:

  1. Loads and inspects the dataset (detects column data-types).
  2. Pre-processes data automatically:
       - Numeric columns   → impute missing values with median.
       - Categorical cols  → one-hot encode.
       - Datetime cols     → extract year / month / day-of-year numeric features.
  3. Performs feature selection via two complementary methods:
       a) Lasso (L1-regularised linear regression)  – linear importance.
       b) Random Forest Regressor                   – non-linear importance.
  4. Keeps features selected by at least one method.
  5. Trains a final Linear Regression model on the selected features.
  6. Reports:
       - Model performance (R², RMSE, MAE on held-out test set).
       - Ranked list of relevant columns.
       - Saves a feature-importance bar chart as feature_importance.png.

Usage
-----
  python col134_feature_selection.py [--csv <path>] [--target <column>]

Defaults
--------
  --csv    elevator_data.csv
  --target COL_134
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Feature selection & regression for COL_134")
    parser.add_argument("--csv",    default="elevator_data.csv",
                        help="Path to the input CSV file (default: elevator_data.csv)")
    parser.add_argument("--target", default="COL_134",
                        help="Name of the target column (default: COL_134)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for testing (default: 0.2)")
    parser.add_argument("--lasso-alpha", type=float, default=0.01,
                        help="Lasso regularisation strength (default: 0.01)")
    parser.add_argument("--rf-trees", type=int, default=200,
                        help="Number of trees for Random Forest (default: 200)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Step 1: Load & inspect
# ---------------------------------------------------------------------------
def load_and_inspect(csv_path: str, target_col: str) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1 – Loading dataset and inspecting column types")
    print("="*60)

    df = pd.read_csv(csv_path)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Detect the type category of every column
    type_map = {}
    for col in df.columns:
        # Try parsing as datetime if dtype is object
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].dropna().iloc[:5])
                type_map[col] = "datetime"
            except Exception:
                type_map[col] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            type_map[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique()
            if n_unique <= 15 and df[col].dtype in (int, "int64", "int32"):
                type_map[col] = "numeric_discrete"
            else:
                type_map[col] = "numeric_continuous"
        else:
            type_map[col] = "other"

    print("\nColumn type summary:")
    print(f"  {'Column':<20}  {'Pandas dtype':<15}  {'Detected category'}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*25}")
    for col, cat in type_map.items():
        marker = " ← TARGET" if col == target_col else ""
        print(f"  {col:<20}  {str(df[col].dtype):<15}  {cat}{marker}")

    if target_col not in df.columns:
        sys.exit(f"\nERROR: Target column '{target_col}' not found in dataset.")

    target_dtype_cat = type_map[target_col]
    print(f"\nTarget column '{target_col}' detected as: {target_dtype_cat}")
    if "numeric" not in target_dtype_cat:
        print(f"WARNING: '{target_col}' does not appear to be numeric. "
              "Regression models require a numeric target. Proceeding anyway.")

    return df, type_map

# ---------------------------------------------------------------------------
# Step 2: Pre-process
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, type_map: dict, target_col: str):
    print("\n" + "="*60)
    print("STEP 2 – Pre-processing features")
    print("="*60)

    df = df.copy()

    # Separate target
    y = pd.to_numeric(df[target_col], errors="coerce")
    df = df.drop(columns=[target_col])
    type_map = {k: v for k, v in type_map.items() if k != target_col}

    feature_frames = []

    # Numeric columns
    num_cols = [c for c, t in type_map.items() if "numeric" in t]
    if num_cols:
        X_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        X_num_imp = pd.DataFrame(imputer.fit_transform(X_num),
                                 columns=num_cols, index=df.index)
        feature_frames.append(X_num_imp)
        print(f"  Numeric columns ({len(num_cols)}): {num_cols}")

    # Categorical columns → one-hot encode
    cat_cols = [c for c, t in type_map.items() if t == "categorical"]
    if cat_cols:
        X_cat = df[cat_cols].astype(str).fillna("MISSING")
        X_cat_ohe = pd.get_dummies(X_cat, prefix=cat_cols, drop_first=False)
        feature_frames.append(X_cat_ohe)
        print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")
        print(f"    → One-hot encoded to {X_cat_ohe.shape[1]} binary columns")

    # Datetime columns → extract numeric features
    dt_cols = [c for c, t in type_map.items() if t == "datetime"]
    if dt_cols:
        dt_frames = []
        for col in dt_cols:
            parsed = pd.to_datetime(df[col], errors="coerce")
            dt_frames.append(pd.DataFrame({
                f"{col}_year":       parsed.dt.year.fillna(0).astype(int),
                f"{col}_month":      parsed.dt.month.fillna(0).astype(int),
                f"{col}_dayofyear":  parsed.dt.dayofyear.fillna(0).astype(int),
            }, index=df.index))
        feature_frames.append(pd.concat(dt_frames, axis=1))
        print(f"  Datetime columns ({len(dt_cols)}): {dt_cols}")
        print(f"    → Extracted year / month / day-of-year numeric features")

    if not feature_frames:
        sys.exit("ERROR: No usable feature columns found after pre-processing.")

    X = pd.concat(feature_frames, axis=1)

    # Drop rows where target is NaN
    valid = y.notna()
    X, y = X[valid], y[valid]

    print(f"\nFeature matrix shape after pre-processing: {X.shape}")
    print(f"Target vector length: {len(y)}")
    return X, y

# ---------------------------------------------------------------------------
# Step 3a: Feature selection via Lasso
# ---------------------------------------------------------------------------
def lasso_selection(X_train, y_train, alpha: float):
    print("\n" + "="*60)
    print("STEP 3a – Lasso (L1) feature selection")
    print("="*60)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  Lasso(alpha=alpha, max_iter=10_000)),
    ])
    pipe.fit(X_train, y_train)

    coef = pipe.named_steps["lasso"].coef_
    lasso_importance = pd.Series(np.abs(coef), index=X_train.columns)
    selected = lasso_importance[lasso_importance > 0].sort_values(ascending=False)

    print(f"\n  Alpha = {alpha}")
    print(f"  Features selected by Lasso: {len(selected)} / {len(X_train.columns)}")
    if len(selected):
        print(f"\n  {'Feature':<30}  |coef|")
        print(f"  {'-'*30}  ------")
        for feat, imp in selected.items():
            print(f"  {feat:<30}  {imp:.6f}")
    return set(selected.index)

# ---------------------------------------------------------------------------
# Step 3b: Feature selection via Random Forest
# ---------------------------------------------------------------------------
def rf_selection(X_train, y_train, n_estimators: int, top_n_pct: float = 0.3):
    print("\n" + "="*60)
    print("STEP 3b – Random Forest feature importance")
    print("="*60)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    rf_importance = rf_importance.sort_values(ascending=False)

    # Select features above the mean importance threshold
    threshold = rf_importance.mean()
    selected = rf_importance[rf_importance >= threshold]

    print(f"\n  Trees = {n_estimators}")
    print(f"  Importance threshold (mean): {threshold:.6f}")
    print(f"  Features selected by Random Forest: {len(selected)} / {len(X_train.columns)}")
    print(f"\n  {'Feature':<30}  Importance")
    print(f"  {'-'*30}  ----------")
    for feat, imp in rf_importance.items():
        marker = " ✓" if feat in selected.index else ""
        print(f"  {feat:<30}  {imp:.6f}{marker}")

    return set(selected.index), rf_importance

# ---------------------------------------------------------------------------
# Step 4: Train final Linear Regression on selected features
# ---------------------------------------------------------------------------
def train_final_model(X_train, X_test, y_train, y_test, selected_features):
    print("\n" + "="*60)
    print("STEP 4 – Training final Linear Regression model")
    print("="*60)

    feat_list = sorted(selected_features)
    print(f"\n  Selected features used for final model ({len(feat_list)}):")
    for f in feat_list:
        print(f"    • {f}")

    X_tr = X_train[feat_list]
    X_te = X_test[feat_list]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression()),
    ])
    pipe.fit(X_tr, y_train)
    y_pred = pipe.predict(X_te)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"\n  Test-set performance")
    print(f"  ─────────────────────────────")
    print(f"  R²   = {r2:.4f}   (1.0 = perfect)")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    return pipe, feat_list, {"R2": r2, "RMSE": rmse, "MAE": mae}

# ---------------------------------------------------------------------------
# Step 5: Save feature-importance chart
# ---------------------------------------------------------------------------
def save_importance_chart(rf_importance: pd.Series, selected_features: set,
                          output_path: str = "feature_importance.png"):
    top = rf_importance.head(20)
    colors = ["#2ecc71" if f in selected_features else "#95a5a6" for f in top.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.45)))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("Feature Importance for COL_134\n"
                 "(green = selected, grey = below threshold)")
    ax.axvline(rf_importance.mean(), color="red", linestyle="--", linewidth=1,
               label=f"Mean threshold = {rf_importance.mean():.4f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  Feature importance chart saved → {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Step 1
    df, type_map = load_and_inspect(args.csv, args.target)

    # Step 2
    X, y = preprocess(df, type_map, args.target)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # Step 3a – Lasso
    lasso_feats = lasso_selection(X_train, y_train, alpha=args.lasso_alpha)

    # Step 3b – Random Forest
    rf_feats, rf_importance = rf_selection(X_train, y_train, n_estimators=args.rf_trees)

    # Union of both methods
    combined_features = lasso_feats | rf_feats
    print("\n" + "="*60)
    print("STEP 3c – Combined feature set (union of Lasso + RF)")
    print("="*60)
    print(f"\n  Features selected by at least one method: {len(combined_features)}")
    for f in sorted(combined_features):
        src = []
        if f in lasso_feats: src.append("Lasso")
        if f in rf_feats:    src.append("RF")
        print(f"    • {f:<30}  [{', '.join(src)}]")

    if not combined_features:
        sys.exit("ERROR: No features selected by either method. "
                 "Try reducing --lasso-alpha or adjusting the dataset.")

    # Step 4
    model, feat_list, metrics = train_final_model(
        X_train, X_test, y_train, y_test, combined_features
    )

    # Step 5
    save_importance_chart(rf_importance, combined_features)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n  Target column  : {args.target}")
    print(f"  Dataset file   : {args.csv}")
    print(f"  Total features : {X.shape[1]}")
    print(f"  Selected feats : {len(feat_list)}")
    print(f"\n  Model          : Linear Regression (on selected features)")
    print(f"  R²             : {metrics['R2']:.4f}")
    print(f"  RMSE           : {metrics['RMSE']:.4f}")
    print(f"  MAE            : {metrics['MAE']:.4f}")
    print(f"\n  The following columns are identified as RELEVANT to {args.target}:")
    for f in feat_list:
        print(f"    ✓ {f}")
    print("\n  Output files:")
    print("    • feature_importance.png  – bar chart of feature importances")
    print()


if __name__ == "__main__":
    main()
