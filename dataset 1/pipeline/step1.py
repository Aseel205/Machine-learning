import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42
N_SPLITS = 5

# Load Data
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]
X_test = test_df.drop(columns=["id"])

# Feature Engineering: BMI
X["BMI"] = X["weight(kg)"] / ((X["height(cm)"] / 100) ** 2)
X_test["BMI"] = X_test["weight(kg)"] / ((X_test["height(cm)"] / 100) ** 2)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Models to evaluate
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "SVM": SVC(probability=True, random_state=SEED),
    "LightGBM": LGBMClassifier(random_state=SEED),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=SEED),
    "CatBoost": CatBoostClassifier(verbose=0, random_seed=SEED)
}

# Cross-validation benchmark
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
results = {}

for name, model in models.items():
    oof_preds = np.zeros(len(X))
    print(f"\nTraining model: {name}")
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
        X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    
    auc = roc_auc_score(y, oof_preds)
    results[name] = auc
    print(f"→ {name} AUC: {auc:.5f}")

# Sort results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\n📊 Model Benchmark Results (AUC):")
for name, score in sorted_results:
    print(f"{name:<20} : {score:.5f}")
