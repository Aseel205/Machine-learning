import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import optuna

SEED = 42
N_SPLITS = 7

# Load preprocessed and scaled data
X_scaled = pd.read_csv("X_scaled.csv")
y = pd.read_csv("dataset/train.csv")["smoking"]

# Compute class imbalance
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# Optuna objective function
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": SEED,
        "n_estimators": 3000,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,  # ✅ Use all CPU cores
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
    }

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, val_idx in kf.split(X_scaled, y):
        X_tr, X_val = X_scaled.iloc[tr_idx], X_scaled.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(100), log_evaluation(0)])

        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

    return np.mean(aucs)

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

# Best result
print("\n🎯 Best AUC:", study.best_value)
print("⚙️ Best Params:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")
