# 2022-07-05
'''
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
#FARE = 2.25 * TRIP_MILES + 0.12 * TRIP_MINUTES + 3.25

^ I want to generate a short python program that will show build a linear regression model to predict taxi fares using pytorch.  Build and train the model, then after showing its efficacy, hardcode the weights of the model based on the formula shown above.

# Follow up
1) Rename the column names to use ALL_CAPS to match the datas source.
2) Remove the scaling of inputs
'''
# chicago_taxi_lr_noscale.py
#
# Predict Chicago taxi fares, then overwrite the model with the
# published linear formula (no input scaling, columns in ALL_CAPS).

import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

# 1 — Load data & rename columns to ALL_CAPS
url = "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
df = pd.read_csv(url)
df.columns = [c.upper() for c in df.columns]        # ALL_CAPS

# 2 — Feature engineering (use minutes instead of seconds)
df["TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60.0

# 3 — Prepare raw (unscaled) tensors
X = df[["TRIP_MILES", "TRIP_MINUTES"]].values.astype(np.float32)
y = df["FARE"].values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

X_train_t = torch.tensor(X_train)
X_test_t  = torch.tensor(X_test)
y_train_t = torch.tensor(y_train)
y_test_t  = torch.tensor(y_test)

# 4 — Define & train a simple linear model (no scaling!)
model = nn.Linear(in_features=2, out_features=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 1_000
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

# 5 — Evaluate learned parameters
with torch.no_grad():
    mse_learned = criterion(model(X_test_t), y_test_t).item()
print(f"Learned weights: {model.weight.data.numpy().ravel()}")
print(f"Learned bias   : {model.bias.data.item():.4f}")
print(f"Test MSE (learned): {mse_learned:.2f}")

# 6 — Hard-code the published formula
with torch.no_grad():
    model.weight.copy_(torch.tensor([[2.25, 0.12]], dtype=torch.float32))
    model.bias.copy_(torch.tensor([3.25], dtype=torch.float32))
    mse_manual = criterion(model(X_test_t), y_test_t).item()

print("\nAfter hard-coding published formula:")
print("Manual weights: [2.25  0.12]")
print("Manual bias   : 3.25")
print(f"Test MSE (manual): {mse_manual:.2f}")
