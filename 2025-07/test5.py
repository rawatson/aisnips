#2025-07-03
'''
Starting with this code, use xgboost to classify the data, reporting accuracy, precision, recall, and graphing the ROC

rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")
rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
]]
'''
# --------------------------------------------------------------
# 1. Setup
# --------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_curve, auc
)
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 2. Load & prepare data
#    (choose ONE of the two options below)
# --------------------------------------------------------------
# --- Option A: direct CSV link (quick & simple) ----------------
url = "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
rice_df = pd.read_csv(url)

# --- Option B (fallback): pull from the UCI repository ---------
# from ucimlrepo import fetch_ucirepo
# rice = fetch_ucirepo(id=545)           # Rice (Cammeo & Osmancik) dataset
# rice_df = pd.concat([rice.data.features, rice.data.targets], axis=1)

# Keep the seven morphology features + label
cols   = [
    "Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length",
    "Eccentricity", "Convex_Area", "Extent", "Class"
]
rice_df = rice_df[cols]

# Numeric-encode the target (Cammeo → 0, Osmancik → 1)
lab    = LabelEncoder()
rice_df["y"] = lab.fit_transform(rice_df["Class"])
X      = rice_df.drop(columns=["Class", "y"])
y      = rice_df["y"]

# --------------------------------------------------------------
# 3. Train/test split
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --------------------------------------------------------------
# 4. Model training
# --------------------------------------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",        # stops warning chatter
    use_label_encoder=False,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

# --------------------------------------------------------------
# 5. Evaluation metrics
# --------------------------------------------------------------
y_pred        = model.predict(X_test)
y_pred_proba  = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score (y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score   (y_test, y_pred)

print("Hold-out performance")
print(f"  • Accuracy : {acc :.4f}")
print(f"  • Precision: {prec:.4f}")
print(f"  • Recall   : {rec :.4f}")

# --------------------------------------------------------------
# 6. ROC curve (one plot, default colours)
# --------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")        # chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title ("ROC – Cammeo vs Osmancik")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
