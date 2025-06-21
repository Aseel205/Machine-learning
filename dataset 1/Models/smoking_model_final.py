# SUPER-OPTIMIZED STACKED ENSEMBLE MODEL FOR 0.92+ AUC

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42
N_SPLITS = 7
np.random.seed(SEED)

# Load data
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
submission_df = pd.read_csv("dataset/sample_submission.csv")

X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]
X_test = test_df.drop(columns=["id"])

# Feature Engineering
for df in [X, X_test]:
    df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["pulse"] = df["systolic"] - df["relaxation"]
    df["AST_ALT"] = df["AST"] / (df["ALT"] + 1e-5)
    df["GTP_ALT"] = df["Gtp"] / (df["ALT"] + 1e-5)
    df["eye_diff"] = df["eyesight(left)"] - df["eyesight(right)"]
    df["log_Gtp"] = np.log1p(df["Gtp"])
    df["log_Chol"] = np.log1p(df["Cholesterol"])
    df["weight_height_ratio"] = df["weight(kg)"] / df["height(cm)"]
    df["systolic_over_BMI"] = df["systolic"] / (df["BMI"] + 1e-5)
    df["systolic_x_BMI"] = df["systolic"] * df["BMI"]
    df["age_x_GTP"] = df["age"] * df["Gtp"]
    df["ALT_to_Chol"] = df["ALT"] / (df["Cholesterol"] + 1e-5)
    df["BMI_bin"] = pd.cut(df["BMI"], bins=5, labels=False)
    df["age_bin"] = pd.cut(df["age"], bins=5, labels=False)
    df["is_obese"] = (df["BMI"] > 30).astype(int)
    df["has_hypertension"] = (df["systolic"] > 140).astype(int)

# Scaling
scaler = QuantileTransformer(output_distribution="normal")
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

scale_pos_weight = (len(y) - sum(y)) / sum(y)

# Define base model parameters
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": SEED,
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "num_leaves": 150,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 10,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "scale_pos_weight": scale_pos_weight
}

xgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "eval_metric": "auc",
    "random_state": SEED,
    "scale_pos_weight": scale_pos_weight
}

cat_params = {
    "iterations": 1500,
    "learning_rate": 0.01,
    "depth": 6,
    "eval_metric": "AUC",
    "verbose": 0,
    "random_seed": SEED,
    "scale_pos_weight": scale_pos_weight
}

# Train models
print("\n🚀 Training base models...")
oof = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100), log_evaluation(0)])
    oof[val_idx, 0] = lgb.predict_proba(X_val)[:, 1]
    test_preds[:, 0] += lgb.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr)
    oof[val_idx, 1] = xgb.predict_proba(X_val)[:, 1]
    test_preds[:, 1] += xgb.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    oof[val_idx, 2] = cat.predict_proba(X_val)[:, 1]
    test_preds[:, 2] += cat.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

# Weighted average ensemble
print("\n🧠 Applying weighted ensemble...")
from scipy.optimize import minimize

def loss_fn(weights):
    blended = weights[0] * oof[:, 0] + weights[1] * oof[:, 1] + weights[2] * oof[:, 2]
    return -roc_auc_score(y, blended)

constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
initial_weights = [0.4, 0.4, 0.2]
res = minimize(loss_fn, initial_weights, method='SLSQP', bounds=[(0,1)]*3, constraints=constraints)
best_weights = res.x
print(f"Optimized Weights: {best_weights}")

final_preds = (
    test_preds[:, 0] * best_weights[0] +
    test_preds[:, 1] * best_weights[1] +
    test_preds[:, 2] * best_weights[2]
)

# Evaluate
auc = roc_auc_score(y, best_weights[0] * oof[:, 0] + best_weights[1] * oof[:, 1] + best_weights[2] * oof[:, 2])
print(f"\n✅ Final Weighted AUC: {auc:.5f}")

submission_df["smoking"] = final_preds
submission_df.to_csv("submission_weighted.csv", index=False)
print("\n📁 Saved submission_weighted.csv")
