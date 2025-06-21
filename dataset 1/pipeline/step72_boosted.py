import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ========== 1. Load and Rank Base Model OOFs ==========
def to_rank(arr):
    return rankdata(arr) / len(arr)

# Load OOFs and apply rank normalization
oof_lgb = to_rank(pd.read_csv("oof_lgb.csv", header=None).values.ravel()).reshape(-1, 1)
oof_xgb = to_rank(pd.read_csv("oof_xgb.csv", header=None).values.ravel()).reshape(-1, 1)
oof_cat = to_rank(pd.read_csv("oof_cat.csv", header=None).values.ravel()).reshape(-1, 1)

# Combine and create meta-features
X_stack_base = np.hstack([oof_lgb, oof_xgb, oof_cat])
X_stack = np.hstack([
    X_stack_base,
    np.var(X_stack_base, axis=1).reshape(-1, 1),
    np.min(X_stack_base, axis=1).reshape(-1, 1),
    np.max(X_stack_base, axis=1).reshape(-1, 1)
])
y_true = pd.read_csv("y.csv", header=None).values.ravel()

# ========== 2. Load Test Set and Process ==========
test_lgb = to_rank(pd.read_csv("test_preds_lgb.csv")["smoking"].values).reshape(-1, 1)
test_xgb = to_rank(pd.read_csv("test_preds_xgb.csv")["smoking"].values).reshape(-1, 1)
test_cat = to_rank(pd.read_csv("test_preds_cat.csv")["smoking"].values).reshape(-1, 1)

X_test_base = np.hstack([test_lgb, test_xgb, test_cat])
X_test_stack = np.hstack([
    X_test_base,
    np.var(X_test_base, axis=1).reshape(-1, 1),
    np.min(X_test_base, axis=1).reshape(-1, 1),
    np.max(X_test_base, axis=1).reshape(-1, 1)
])
test_ids = pd.read_csv("test_preds_lgb.csv")["id"]

# ========== 3. Train Meta-Model (LGBM) with CV ==========
cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)
scores = []
test_preds = np.zeros(X_test_stack.shape[0])

for fold, (train_idx, val_idx) in enumerate(cv.split(X_stack, y_true)):
    X_train, X_val = X_stack[train_idx], X_stack[val_idx]
    y_train, y_val = y_true[train_idx], y_true[val_idx]

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        random_state=SEED + fold,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    scores.append(roc_auc_score(y_val, val_pred))

    test_preds += model.predict_proba(X_test_stack)[:, 1] / cv.n_splits

print(f"✅ Meta-model CV AUCs: {['%.5f' % s for s in scores]}")
print(f"🎯 Mean AUC: {np.mean(scores):.5f}")

# ========== 4. Save Submission ==========
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds
})
submission.to_csv("submission_final.csv", index=False)
print("✅ submission_final.csv saved.")
