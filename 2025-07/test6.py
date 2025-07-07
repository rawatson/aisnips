#2025-07-03
'''
Starting with this code, I want to evaluate the performance of 3-5 different classification methods, focusing on the more performant ones.  Write code that will build 3-5 models, then compute the ROC for each and plot it.  You can use sklearn, xgboost, pandas, pytorch, or any other common python ML libraries

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

Follow up:
Suggest 3 other models that could be used here

Update the code to add in these 3 models
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# ── Scikit-learn core ──────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ── Gradient-boosting libraries ────────────────────────────────────────────────
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier            # pip install lightgbm
from catboost import CatBoostClassifier        # pip install catboost

# ── 1. Load & prepare data ─────────────────────────────────────────────────────
rice_dataset_raw = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
)
rice_dataset = rice_dataset_raw[
    [
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Eccentricity",
        "Convex_Area",
        "Extent",
        "Class",
    ]
]

label_encoder = LabelEncoder()
rice_dataset["Class"] = label_encoder.fit_transform(rice_dataset["Class"])
X = rice_dataset.drop(columns=["Class"])
y = rice_dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 2. Define models ───────────────────────────────────────────────────────────
models = {
    # baselines
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
    "SVM (RBF kernel)": SVC(probability=True, kernel="rbf", C=2.0, gamma="scale"),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
    ),
    # NEW additions
    "LightGBM": LGBMClassifier(
        random_state=42, n_estimators=400, learning_rate=0.05
    ),
    "CatBoost": CatBoostClassifier(
        verbose=0, random_seed=42, iterations=500, depth=6
    ),
    "Histogram GBDT": HistGradientBoostingClassifier(
        max_depth=None, learning_rate=0.05
    ),
}

# ── 3. Train, collect ROC & confusion matrices ────────────────────────────────
plt.figure(figsize=(10, 7))
conf_mats = {}

for name, model in models.items():
    needs_scaling = "SVM" in name or "k-NN" in name
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_proba = model.predict_proba(Xte)[:, 1]
    y_pred = model.predict(Xte)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    conf_mats[name] = confusion_matrix(y_test, y_pred)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Rice Variety Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# ── 4. Confusion-matrix report & visuals ───────────────────────────────────────
print("\nConfusion Matrices (rows = true class, cols = predicted class):")
for name, cm in conf_mats.items():
    print(f"\n{name}:\n{cm}")

# Dynamic grid: 3 columns
n_models = len(conf_mats)
cols = 3
rows = ceil(n_models / cols)
plt.figure(figsize=(5 * cols, 4 * rows))

for idx, (name, cm) in enumerate(conf_mats.items(), 1):
    ax = plt.subplot(rows, cols, idx)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()
