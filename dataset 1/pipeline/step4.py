# ✅ FINAL LGBM MODEL USING OPTUNA TUNED PARAMS (7-FOLD CV)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

SEED = 42
N_SPLITS = 7

# Load preprocessed data
X_scaled = pd.read_csv("X_scaled.csv")
y = pd.read_csv("dataset/train.csv")["smoking"]
X_test_scaled = pd.read_csv("X_test_scaled.csv")  # Make sure this exists
submission_df = pd.read_csv("dataset/sample_submission.csv")

# Final tuned params from Optuna
best_params = {
    "learning_rate": 0.01603290932347039,
    "max_depth": 4,
    "num_leaves": 30,
    "feature_fraction": 0.7706008960014699,
    "bagging_fraction": 0.7039165116665395,
    "bagging_freq": 1,
    "min_child_samples": 100,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": SEED,
    "n_estimators": 3000,
    "scale_pos_weight": (len(y) - sum(y)) / sum(y)
}

# Train with full 7-fold CV and evaluate
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X_scaled))
test_preds = np.zeros(len(X_test_scaled))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = LGBMClassifier(**best_params, n_jobs=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              callbacks=[early_stopping(100), log_evaluation(0)])

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    test_preds += model.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

auc = roc_auc_score(y, oof_preds)
print(f"\n✅ FINAL LGBM CV AUC: {auc:.5f}")

# Save submission
submission_df["smoking"] = test_preds
submission_df.to_csv("submission_lgbm_tuned.csv", index=False)
print("\n📁 Saved: submission_lgbm_tuned.csv")
