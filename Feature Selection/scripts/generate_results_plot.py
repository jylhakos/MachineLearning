"""
generate_results_plot.py

Generates a demonstration result plot for the COL_134 regression pipeline.
Produces a four-panel figure saved to outputs/col134_results_plot.png.

Panels:
  1. Predicted vs Actual scatter plot with identity line
  2. Residuals vs Predicted values
  3. Feature importance bar chart (top 10 features)
  4. Five-method comparison bar chart (Test R2)

Run inside the activated virtual environment:
    source venv/bin/activate
    python scripts/generate_results_plot.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducible demo dataset
# ---------------------------------------------------------------------------
SEED = 42
rng  = np.random.default_rng(SEED)
N    = 500

data = {f"COL_{i}": rng.normal(size=N) for i in range(1, 31)}
df   = pd.DataFrame(data)

# COL_134 is a linear combination of 3 columns plus noise
df["COL_134"] = (
    2.5  * df["COL_1"]
    - 1.8 * df["COL_2"]
    + 0.9 * df["COL_5"]
    + 0.4 * df["COL_8"]
    + rng.normal(scale=0.5, size=N)
)

TARGET = "COL_134"
y      = df[TARGET]
X_raw  = df.drop(columns=[TARGET]).select_dtypes(include="number")

# Shared preprocessing
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X = pd.DataFrame(
    scaler.fit_transform(imputer.fit_transform(X_raw)),
    columns=X_raw.columns,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ---------------------------------------------------------------------------
# Main model: Random Forest + Lasso dual selection -> Linear Regression
# ---------------------------------------------------------------------------
lasso = Lasso(alpha=0.01, max_iter=10000, random_state=SEED)
lasso.fit(X_train, y_train)
lasso_mask = lasso.coef_ != 0

rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rf_sel  = SelectFromModel(rf, threshold="mean", prefit=True)
rf_mask = rf_sel.get_support()

union_mask     = lasso_mask | rf_mask
feature_names  = X.columns[union_mask].tolist()

X_train_sel = X_train.iloc[:, union_mask]
X_test_sel  = X_test.iloc[:, union_mask]

final_model = LinearRegression()
final_model.fit(X_train_sel, y_train)
y_pred = final_model.predict(X_test_sel)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
residuals = y_test.values - y_pred

# Feature importances from Random Forest (union features only)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances[feature_names].nlargest(10)

# ---------------------------------------------------------------------------
# Five-method comparison
# ---------------------------------------------------------------------------
kf          = KFold(n_splits=5, shuffle=True, random_state=SEED)
eval_model  = LinearRegression()
method_r2   = {}

def quick_eval(mask):
    if mask.sum() == 0:
        return 0.0
    cv = cross_val_score(eval_model, X_train.iloc[:, mask], y_train,
                         cv=kf, scoring="r2")
    eval_model.fit(X_train.iloc[:, mask], y_train)
    return r2_score(y_test, eval_model.predict(X_test.iloc[:, mask]))

method_r2["Lasso"]              = quick_eval(lasso_mask)
method_r2["Random Forest"]      = quick_eval(rf_mask)
method_r2["Lasso + RF (union)"] = quick_eval(union_mask)

gb     = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED)
gb.fit(X_train, y_train)
gb_sel = SelectFromModel(gb, threshold="mean", prefit=True)
method_r2["Gradient Boosting"]  = quick_eval(gb_sel.get_support())

if HAS_XGB:
    xgb     = XGBRegressor(n_estimators=200, learning_rate=0.05,
                            random_state=SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_sel = SelectFromModel(xgb, threshold="median", prefit=True)
    method_r2["XGBoost"] = quick_eval(xgb_sel.get_support())
else:
    mi_scores = mutual_info_regression(X_train, y_train, random_state=SEED)
    mi_mask   = mi_scores > np.mean(mi_scores)
    method_r2["Mutual Info"] = quick_eval(mi_mask)

methods_sorted = sorted(method_r2.items(), key=lambda x: x[1], reverse=True)
m_names = [m[0] for m in methods_sorted]
m_vals  = [m[1] for m in methods_sorted]

# ---------------------------------------------------------------------------
# Four-panel figure
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

ACCENT = "#2980b9"
WARM   = "#e67e22"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
GRAY   = "#7f8c8d"

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#fafafa")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ---- Panel 1: Predicted vs Actual ----------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#f4f9fd")
ax1.scatter(y_test, y_pred, alpha=0.55, s=35, color=ACCENT, edgecolors="none",
            label="Samples")
lim_min = min(y_test.min(), y_pred.min()) - 0.5
lim_max = max(y_test.max(), y_pred.max()) + 0.5
ax1.plot([lim_min, lim_max], [lim_min, lim_max], color=WARM,
         linewidth=2, linestyle="--", label="Perfect prediction")
ax1.set_xlabel("Actual COL_134", fontsize=11)
ax1.set_ylabel("Predicted COL_134", fontsize=11)
ax1.set_title("Predicted vs Actual (COL_134)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
stats_text = (f"R$^2$ = {r2:.4f}\n"
              f"RMSE = {rmse:.4f}\n"
              f"MAE  = {mae:.4f}\n"
              f"n features = {union_mask.sum()}")
ax1.text(0.04, 0.96, stats_text, transform=ax1.transAxes,
         fontsize=9, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
ax1.grid(True, alpha=0.3)

# ---- Panel 2: Residuals vs Predicted -------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#f4f9fd")
ax2.scatter(y_pred, residuals, alpha=0.55, s=35, color=GREEN, edgecolors="none")
ax2.axhline(0, color=WARM, linewidth=2, linestyle="--")
ax2.set_xlabel("Predicted COL_134", fontsize=11)
ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
ax2.set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.text(0.04, 0.96,
         f"Residual std = {residuals.std():.4f}",
         transform=ax2.transAxes, fontsize=9, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

# ---- Panel 3: Feature Importance (top 10) --------------------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#fffdf4")
colors_bar = [ACCENT if f in ["COL_1", "COL_2", "COL_5", "COL_8"] else GRAY
              for f in top_features.index]
ax3.barh(top_features.index[::-1], top_features.values[::-1],
         color=colors_bar[::-1], edgecolor="none")
ax3.set_xlabel("Random Forest Importance", fontsize=11)
ax3.set_title("Top 10 Selected Features\n(blue = true signal features)",
              fontsize=12, fontweight="bold")
ax3.grid(True, axis="x", alpha=0.3)
for i, (name, val) in enumerate(zip(top_features.index[::-1],
                                     top_features.values[::-1])):
    ax3.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8, color="#333")

# ---- Panel 4: Five-method R2 comparison ----------------------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#fdf4ff")
bar_colors = [PURPLE if v == max(m_vals) else ACCENT for v in m_vals]
bars = ax4.bar(m_names, m_vals, color=bar_colors, edgecolor="none", width=0.6)
ax4.set_ylabel("Test R$^2$", fontsize=11)
ax4.set_title("Feature Selection Method Comparison\n(same final model: Linear Regression)",
              fontsize=12, fontweight="bold")
ax4.set_ylim(0, 1.05)
ax4.axhline(0.90, color=GREEN,  linewidth=1.5, linestyle=":", alpha=0.7, label="Excellent (0.90)")
ax4.axhline(0.75, color=WARM,   linewidth=1.5, linestyle=":", alpha=0.7, label="Good (0.75)")
ax4.legend(fontsize=8, loc="lower right")
ax4.tick_params(axis="x", labelrotation=20, labelsize=9)
ax4.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, m_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

fig.suptitle(
    "COL_134 Regression Pipeline: Model Results\n"
    "Dual Feature Selection (Lasso + Random Forest Union)  |  Final Model: Linear Regression",
    fontsize=13, fontweight="bold", y=0.98
)

out_path = "outputs/col134_results_plot.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Plot saved to {out_path}")

# Print a brief summary to stdout
print("\n=== COL_134 Pipeline Result Summary ===")
print(f"  Features in input  : {X.shape[1]}")
print(f"  Features selected  : {union_mask.sum()}")
print(f"  Selected columns   : {feature_names}")
print(f"  Test R2            : {r2:.4f}")
print(f"  Test RMSE          : {rmse:.4f}")
print(f"  Test MAE           : {mae:.4f}")
print("\n  Method comparison (Test R2):")
for name, val in methods_sorted:
    marker = " <-- best" if val == max(m_vals) else ""
    print(f"    {name:25s}  {val:.4f}{marker}")
