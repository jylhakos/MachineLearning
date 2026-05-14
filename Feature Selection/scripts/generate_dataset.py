"""
generate_dataset.py
-------------------
Generates a synthetic elevator-component dataset and saves it as
elevator_data.csv.  The file is self-contained and does not require
any installed packages beyond the standard library + numpy + pandas.

Column layout
-------------
  COL_134   : target – a continuous measurement of an elevator component
               (e.g. vibration amplitude, motor current, wear index).
               It is a *linear* combination of a few "signal" columns
               plus Gaussian noise.

  COL_001   : integer category  (e.g. floor zone)           – relevant
  COL_045   : float, load weight in kg                      – relevant
  COL_067   : float, speed in m/s                           – relevant
  COL_089   : float, motor temperature in °C                – relevant
  COL_102   : integer, number of daily trips                – relevant
  COL_110   : float, door open/close time in seconds        – relevant (weakly)
  COL_120   : float, pure noise column A                    – NOT relevant
  COL_125   : float, pure noise column B                    – NOT relevant
  COL_130   : float, pure noise column C                    – NOT relevant
  COL_140   : string category (maintenance status)          – included to
               demonstrate mixed-type detection & encoding
  COL_150   : datetime (last service timestamp)             – included to
               demonstrate datetime-type detection
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N = 500   # number of samples

# --- signal columns (known to affect COL_134) ---
col_001 = RNG.integers(1, 6, size=N).astype(float)          # floor zone 1-5
col_045 = RNG.uniform(50, 800, size=N)                       # load weight kg
col_067 = RNG.uniform(0.5, 3.0, size=N)                     # speed m/s
col_089 = RNG.uniform(20, 90, size=N)                        # motor temp °C
col_102 = RNG.integers(10, 200, size=N).astype(float)        # daily trips
col_110 = RNG.uniform(1.0, 6.0, size=N)                     # door time s

# --- irrelevant / noise columns ---
col_120 = RNG.normal(0, 1, size=N)
col_125 = RNG.uniform(-5, 5, size=N)
col_130 = RNG.integers(0, 100, size=N).astype(float)

# --- target COL_134 (linear relationship + noise) ---
noise = RNG.normal(0, 2, size=N)
col_134 = (
    0.8  * col_001
    + 0.05 * col_045
    + 3.2  * col_067
    + 0.4  * col_089
    + 0.02 * col_102
    + 0.6  * col_110
    + noise
)

# --- categorical column (maintenance status) ---
statuses = RNG.choice(["OK", "WARNING", "CRITICAL"], size=N, p=[0.7, 0.2, 0.1])

# --- datetime column (last service) ---
base_date = pd.Timestamp("2024-01-01")
days_offset = RNG.integers(0, 365, size=N)
last_service = [base_date + pd.Timedelta(days=int(d)) for d in days_offset]

df = pd.DataFrame({
    "COL_001": col_001.astype(int),
    "COL_045": col_045,
    "COL_067": col_067,
    "COL_089": col_089,
    "COL_102": col_102.astype(int),
    "COL_110": col_110,
    "COL_120": col_120,
    "COL_125": col_125,
    "COL_130": col_130.astype(int),
    "COL_134": col_134,          # target
    "COL_140": statuses,
    "COL_150": last_service,
})

# Round floats for readability
float_cols = ["COL_045", "COL_067", "COL_089", "COL_110", "COL_120", "COL_125", "COL_134"]
df[float_cols] = df[float_cols].round(4)

df.to_csv("elevator_data.csv", index=False)
print(f"Dataset saved → elevator_data.csv  ({N} rows, {df.shape[1]} columns)")
print(df.dtypes)
print(df.head(3))
