#2025-07-06
'''
Generate a small snippet of code that manually trains and tests a locally weighted linear regression, visualizing the results and a comparison with a standard linear regression using plotly
'''
import numpy as np
import matplotlib.pyplot as plt

# 1. synthetic data -------------------------------------------------
np.random.seed(0)
n = 40
x_raw = np.sort(np.random.uniform(0, 10, n))
y      = 3 + 0.5*x_raw + np.sin(x_raw) + np.random.normal(0, 0.5, n)

X = np.vstack([np.ones_like(x_raw), x_raw]).T          # add intercept

# 2. ordinary least squares ----------------------------------------
theta_lr = np.linalg.inv(X.T @ X) @ (X.T @ y)
print(theta_lr)

# 3. locally weighted LR helper ------------------------------------
def lwr_predict(x0, Xtr, ytr, tau=0.8):
    w      = np.exp(-((Xtr[:, 1] - x0)**2) / (2*tau**2))
    W      = np.diag(w)
    theta  = np.linalg.pinv(Xtr.T @ W @ Xtr) @ (Xtr.T @ W @ ytr)
    return float(theta @ np.array([1.0, x0]))

# predictions on a grid
grid = np.linspace(0, 10, 300)
y_lr  = theta_lr[0] + theta_lr[1]*grid
y_lwr = np.array([lwr_predict(x0, X, y, tau=0.8) for x0 in grid])

# 4. visual comparison ---------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(x_raw, y, alpha=0.7, label="training data")
plt.plot(grid, y_lr,  linewidth=2, label="standard LR")
plt.plot(grid, y_lwr, linewidth=2, label="locally-weighted LR")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression vs. Locally-Weighted Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()
