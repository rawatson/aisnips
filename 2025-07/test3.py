#2025-07-02
# I generated this because Google's ML crash course claimed the taxi fare was given by a formula... but it definitely wasn't
'''
Add a third step to this script.  Using the hardcoded formula, generate a new set of training and test data that has the labels derived exactly from the hand coded formula.  Then, re-run the training and show the learned weights

(contents of test2)
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --------------------------------------------------------------------------
# 1. Load data --------------------------------------------------------------
# --------------------------------------------------------------------------
URL = "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
df  = pd.read_csv(URL)

# Feature and target selection (column names already ALL_CAPS)
features = ["TRIP_MILES", "TRIP_SECONDS"]   # seconds, not minutes
target   = "FARE"

X = df[features]
y = df[target]

# Hold-out split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# --------------------------------------------------------------------------
# 2. Baseline: train & evaluate on real fares -------------------------------
# --------------------------------------------------------------------------
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print("=== Baseline model (real labels) ===")
print("Learned weights:")
print(f"  intercept      : {linreg.intercept_:.4f}")
for name, coef in zip(features, linreg.coef_):
    print(f"  {name.lower():>12}: {coef:.4f}")

y_pred = linreg.predict(X_test)
print("\nFitted-model performance:")
print(f"  R²  : {r2_score(y_test, y_pred):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, y_pred):.2f} dollars")
print(f"  MSE : {mean_squared_error(y_test, y_pred):.2f} dollars")

# --------------------------------------------------------------------------
# 3. Synthetic-label experiment ---------------------------------------------
#    • Create *new* targets with the rule-of-thumb formula
#    • Retrain and show the learned weights
# --------------------------------------------------------------------------
def rule_of_thumb(miles, seconds):
    """
    Hand-coded fare: 2.25 * miles + 0.12 * minutes + 3.25
    Dataset stores trip duration in seconds, so convert to minutes.
    """
    return 2.25 * miles + 0.12 * (seconds / 60.0) + 3.25

# Build deterministic labels for both splits
y_train_synth = rule_of_thumb(X_train["TRIP_MILES"], X_train["TRIP_SECONDS"])
y_test_synth  = rule_of_thumb(X_test["TRIP_MILES"],  X_test["TRIP_SECONDS"])

linreg_synth = LinearRegression()
linreg_synth.fit(X_train, y_train_synth)

print("\n=== Synthetic-label model (should recover hand-coded weights) ===")
print("Learned weights:")
print(f"  intercept      : {linreg_synth.intercept_:.4f}")
for name, coef in zip(features, linreg_synth.coef_):
    print(f"  {name.lower():>12}: {coef:.4f}")

y_pred_synth = linreg_synth.predict(X_test)
print("\nSynthetic-label model performance (against synthetic labels):")
print(f"  R²  : {r2_score(y_test_synth, y_pred_synth):.4f}")
print(f"  MAE : {mean_absolute_error(y_test_synth, y_pred_synth):.6f} dollars")
print(f"  MSE : {mean_squared_error(y_test_synth, y_pred_synth):.6f} dollars")

# --------------------------------------------------------------------------
# 4. Hard-coded rule-of-thumb performance (real labels) ---------------------
# --------------------------------------------------------------------------
y_rule = rule_of_thumb(X_test["TRIP_MILES"], X_test["TRIP_SECONDS"])

print("\n=== Rule-of-thumb, hard-coded weights (real labels) ===")
print(f"  R²  : {r2_score(y_test, y_rule):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, y_rule):.2f} dollars")
print(f"  MSE : {mean_squared_error(y_test, y_rule):.2f} dollars")
