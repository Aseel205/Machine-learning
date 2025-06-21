import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")

# ======================
# 0. SET REPRODUCIBILITY
# ======================
SEED = 42
np.random.seed(SEED)

# ======================
# 1. LOAD AND PREP DATA
# ======================
oof_lgb = pd.read_csv("oof_lgb.csv").values
oof_xgb = pd.read_csv("oof_xgb.csv").values
oof_cat = pd.read_csv("oof_cat.csv").values
X_stack = np.hstack([oof_lgb, oof_xgb, oof_cat])
y_true = pd.read_csv("y.csv").values.ravel()

test_data = pd.read_csv("test_preds_lgb.csv")
test_ids = test_data["id"]
X_test_stack = np.hstack([
    test_data.drop(columns=["id"]).values,
    pd.read_csv("test_preds_xgb.csv").drop(columns=["id"]).values,
    pd.read_csv("test_preds_cat.csv").drop(columns=["id"]).values
])

# ======================
# 2. OPTIMIZE META-MODEL
# ======================
def objective(trial):
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])

    # Conditional penalty based on solver
    if solver == "liblinear":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    else:  # lbfgs
        penalty = "l2"  # lbfgs supports only l2 or None

    params = {
        "solver": solver,
        "penalty": penalty,
        "Cs": trial.suggest_int("Cs", 3, 10),
        "max_iter": trial.suggest_int("max_iter", 500, 2000),
    }

    model = LogisticRegressionCV(**params, cv=5, random_state=SEED, scoring="roc_auc")
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, val_idx in cv.split(X_stack, y_true):
        X_train, X_val = X_stack[train_idx], X_stack[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    return np.mean(scores)

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
study.optimize(objective, n_trials=20)

# Fit best model
best_meta_model = LogisticRegressionCV(**study.best_params, cv=5, random_state=SEED, scoring="roc_auc")
best_meta_model.fit(X_stack, y_true)

# ======================
# 3. GENERATE SUBMISSION
# ======================
final_test_preds = best_meta_model.predict_proba(X_test_stack)[:, 1]
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": final_test_preds
})
submission.to_csv("submission_final.csv", index=False)
print("✅ Submission saved with IDs: 'submission_final.csv'")
print(submission.head())
