# ✅ XGBoost with Optuna + OOF for Stacking

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna

SEED = 42
N_SPLITS = 7

# Load preprocessed data
X_scaled = pd.read_csv("X_scaled.csv")
y = pd.read_csv("dataset/train.csv")["smoking"]
X_test_scaled = pd.read_csv("X_test_scaled.csv")

# Compute class imbalance
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# Optuna objective
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": scale_pos_weight,
        "random_state": SEED,
        "n_jobs": -1,
    }

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, val_idx in kf.split(X_scaled, y):
        X_tr, X_val = X_scaled.iloc[tr_idx], X_scaled.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = XGBClassifier(**params, use_label_encoder=False)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=False)

        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

    return np.mean(aucs)

# Run Optuna tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

print("\n🎯 Best AUC:", study.best_value)
print("⚙️ Best Params:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

# Save best params for later
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": scale_pos_weight,
    "random_state": SEED,
    "n_estimators": 3000,
    "use_label_encoder": False,
    "n_jobs": -1
})

# OOF predictions for stacking
oof_preds = np.zeros(len(X_scaled))
test_preds = np.zeros(len(X_test_scaled))

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              early_stopping_rounds=100,
              verbose=False)

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    test_preds += model.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

# Save OOF and test predictions
pd.DataFrame({"xgb_oof": oof_preds}).to_csv("oof_xgb.csv", index=False)
pd.DataFrame({"xgb_test": test_preds}).to_csv("test_preds_xgb.csv", index=False)
print("\n📁 Saved: oof_xgb.csv and test_preds_xgb.csv")
