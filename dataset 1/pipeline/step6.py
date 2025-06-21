# ✅ CatBoost with GridSearch + OOF for Stacking

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from itertools import product

SEED = 42
N_SPLITS = 7

# Load data
X_scaled = pd.read_csv("X_scaled.csv")
y = pd.read_csv("dataset/train.csv")["smoking"]
X_test_scaled = pd.read_csv("X_test_scaled.csv")

# Grid search params
param_grid = {
    "learning_rate": [0.01, 0.03],
    "depth": [4, 6],
    "l2_leaf_reg": [1, 3, 5]
}

best_score = 0
best_params = None

for lr, depth, l2 in product(param_grid["learning_rate"], param_grid["depth"], param_grid["l2_leaf_reg"]):
    aucs = []
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for train_idx, valid_idx in kf.split(X_scaled, y):
        X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2,
            iterations=3000,
            eval_metric="AUC",
            random_seed=SEED,
            verbose=0,
        )

        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100)
        preds = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y_valid, preds))

    mean_auc = np.mean(aucs)
    print(f"Params: lr={lr}, depth={depth}, l2={l2} => AUC={mean_auc:.5f}")
    if mean_auc > best_score:
        best_score = mean_auc
        best_params = {"learning_rate": lr, "depth": depth, "l2_leaf_reg": l2}

print("\n🎯 Best CatBoost AUC:", best_score)
print("⚙️ Best Params:", best_params)

# Train final OOF + test predictions
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X_scaled))
test_preds = np.zeros(len(X_test_scaled))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostClassifier(
        **best_params,
        iterations=3000,
        eval_metric="AUC",
        random_seed=SEED,
        verbose=0
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100)

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    test_preds += model.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

# Save OOF and test predictions
pd.DataFrame({"cat_oof": oof_preds}).to_csv("oof_cat.csv", index=False)
pd.DataFrame({"cat_test": test_preds}).to_csv("test_preds_cat.csv", index=False)
print("\n📁 Saved: oof_cat.csv and test_preds_cat.csv")
