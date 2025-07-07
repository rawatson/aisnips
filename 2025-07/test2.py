# 2025-07-02
'''
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
#FARE = 2.25 * TRIP_MILES + 0.12 * TRIP_MINUTES + 3.25

^ I want to generate a short python program that will show build a linear regression model to predict taxi fares using sklearn.  Build and train the model, then after showing its efficacy, hardcode the weights of the model based on the formula shown above.  Note the column names in the data are in all caps
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. Load data --------------------------------------------------------------
URL = "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
df  = pd.read_csv(URL)

# Keep only the features we need (column names are already ALL_CAPS)
features = ["TRIP_MILES", "TRIP_SECONDS"]
target   = "FARE"
X = df[features]
y = df[target]

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. Train model ------------------------------------------------------------
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print("Learned weights")
print(f"  intercept: {linreg.intercept_:.4f}")
for name, coef in zip(features, linreg.coef_):
    print(f"  {name.lower():>12}: {coef:.4f}")

# 3. Evaluate fitted model --------------------------------------------------
y_pred = linreg.predict(X_test)
print("\nFitted-model performance")
print(f"  R²  : {r2_score(y_test, y_pred):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, y_pred):.2f} dollars")
print(f"  MSE : {mean_squared_error(y_test, y_pred):.2f} dollars")


# 4. Hard-coded “rule-of-thumb” model ---------------------------------------
y_rule = (
    2.25 * X_test["TRIP_MILES"]
  + 0.12 / 60 *  X_test["TRIP_SECONDS"]
  + 3.25
)

print("\nRule-of-thumb performance (hard-coded weights)")
print(f"  R²  : {r2_score(y_test, y_rule):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, y_rule):.2f} dollars")
print(f"  MSE : {mean_squared_error(y_test, y_rule):.2f} dollars")

