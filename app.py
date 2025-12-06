# train_model_v5.1.py — Auto Model Selection (Premium Version)
# Built for Sachin Ravi — CHD Predictor v5.1
# This script:
# - Loads & cleans Framingham dataset
# - Trains 6 ML models
# - Picks the best accuracy model
# - Saves artifacts for app.py (v5.0 / v5.1)

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# Try optional models
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False


# --------------------------
# Paths
# --------------------------
ROOT = Path.cwd()
CSV_PATH = ROOT / "framingham.csv"

MODEL_PATH = ROOT / "best_heart_chd_model.joblib"
SCALER_PATH = ROOT / "scaler_chd.joblib"
FEATURE_PATH = ROOT / "feature_order.json"

RANDOM_STATE = 42


# --------------------------
# Load + clean dataset
# --------------------------
print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

df = df.replace(["NA", "?", " ", ""], np.nan)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median(numeric_only=True))

target = "TenYearCHD"
X = df.drop(columns=[target])
y = df[target].astype(int)

feature_names = list(X.columns)


# --------------------------
# Split + scale
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# --------------------------
# Model candidates
# --------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced"
    )
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

if LGB_AVAILABLE:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )


# --------------------------
# Train + Evaluate
# --------------------------
results = {}

print("\nTraining models...\n")

for name, model in models.items():
    print(f"Training: {name}...")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f" → Accuracy = {acc:.4f}\n")


# --------------------------
# Select Best Model
# --------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_accuracy = results[best_model_name]

print("========================================")
print(f" BEST MODEL SELECTED: {best_model_name}")
print(f" BEST ACCURACY: {best_accuracy:.4f}")
print("========================================\n")


# --------------------------
# Save artifacts
# --------------------------
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

with open(FEATURE_PATH, "w") as f:
    json.dump(feature_names, f, indent=2)

print("Artifacts saved:")
print(f" - Model → {MODEL_PATH}")
print(f" - Scaler → {SCALER_PATH}")
print(f" - Feature order → {FEATURE_PATH}")

print("\nTraining complete! 🎉")
