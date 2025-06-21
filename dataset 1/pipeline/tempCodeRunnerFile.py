# ✅ XGBoost with Optuna + OOF for Stacking

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna

SEED = 42
N_SPLITS = 7

# Load data
X_scaled = pd.read_csv("X_scaled.csv")
y = pd.read_csv("y.csv")["smoking"] if "smoking" in pd.read_csv("y.csv").columns else pd.read_csv("y.csv").squeeze()
X_test_scaled = pd.read_csv("X_test_scaled.csv")

# Handle imbalance
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# Optuna objective
def objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "scale_pos_weight": scale_pos_weight,
        "random_state": SEED,
        "n_jobs": -1,
        "use_label_encoder": False,
        "verbosity": 0
    }

    aucs = []
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for train_idx, valid_idx in kf.split(X_scaled, y):
        X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            eval_metric="auc"
        )

        preds = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y_valid, preds))

    return np.mean(aucs)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

print("\n🎯 Best XGBoost AUC:", study.best_value)
print("⚙️ Best Params:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

# Update best params
best_params = study.best_params
best_params.update({
    "n_estimators": 3000,
    "random_state": SEED,
    "n_jobs": -1,
    "scale_pos_weight": scale_pos_weight,
    "use_label_encoder": False,
    "verbosity": 0
})

# OOF & Test Predictions
oof_preds = np.zeros(len(X_scaled))
test_preds = np.zeros(len(X_test_scaled))

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nFold {fold + 1}/{N_SPLITS}")
    X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        eval_metric="auc"
    )

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    test_preds += model.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

# Save predictions
pd.DataFrame({"xgb_oof": oof_preds}).to_csv("oof_xgb.csv", index=False)
pd.DataFrame({"xgb_test": test_preds}).to_csv("test_preds_xgb.csv", index=False)
print("\n📁 Saved: oof_xgb.csv and test_preds_xgb.csv")
