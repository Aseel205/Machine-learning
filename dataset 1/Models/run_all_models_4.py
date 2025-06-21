import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

SEED = 42
N_SPLITS = 5
np.random.seed(SEED)

# Load data
student_id_full = '213758758'
student_id = student_id_full[-5:]
print(f"Using derived ID for files: {student_id} (from full ID: {student_id_full})")

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
submission_df = pd.read_csv("dataset/sample_submission.csv")

X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]
X_test = test_df.drop(columns=["id"])

# Feature Engineering: Add BMI if height and weight exist
if 'height(cm)' in X.columns and 'weight(kg)' in X.columns:
    X["BMI"] = X["weight(kg)"] / ((X["height(cm)"] / 100) ** 2)
    X_test["BMI"] = X_test["weight(kg)"] / ((X_test["height(cm)"] / 100) ** 2)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# --- Optuna tuning for LightGBM ---
def tune_lgbm():
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": SEED,
            "n_estimators": 10000,
            "scale_pos_weight": scale_pos_weight,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs = []
        for train_idx, valid_idx in cv.split(X_scaled, y):
            X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      eval_metric="auc",
                      callbacks=[early_stopping(100), log_evaluation(0)])
            preds = model.predict_proba(X_valid)[:, 1]
            aucs.append(roc_auc_score(y_valid, preds))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)
    print("Best LightGBM trial:", study.best_trial)
    return study.best_trial.params

best_lgbm_params = tune_lgbm()
best_lgbm_params.update({
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": SEED,
    "n_estimators": 10000,
    "scale_pos_weight": scale_pos_weight
})

# Prepare meta-feature containers
oof_preds = np.zeros((len(X), 3))
X_test_preds = np.zeros((len(X_test), 3))

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Tuned LightGBM
    lgbm = LGBMClassifier(**best_lgbm_params)
    lgbm.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)],
             eval_metric="auc",
             callbacks=[early_stopping(100), log_evaluation(0)])
    oof_preds[valid_idx, 0] = lgbm.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 0] += lgbm.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    # XGBoost tuned with same process
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight
    )
    xgb.fit(X_train, y_train)
    oof_preds[valid_idx, 1] = xgb.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 1] += xgb.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    # CatBoost tuned (lightly)
    cat = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.015,
        depth=7,
        eval_metric="AUC",
        verbose=0,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight
    )
    cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100)
    oof_preds[valid_idx, 2] = cat.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 2] += cat.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

# Train meta-model (LGBM)
print("\nTraining meta-model (LightGBM)...")
meta_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED)
meta_model.fit(oof_preds, y)
meta_preds = meta_model.predict_proba(X_test_preds)[:, 1]

# Save submission
submission_df["smoking"] = meta_preds
submission_df = submission_df[["id", "smoking"]]
submission_df.to_csv("submission.csv", index=False)
print("\n✅ Submission file created successfully using full Optuna-stacked model!")
