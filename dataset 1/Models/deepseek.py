import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from xgboost.callback import EarlyStopping

SEED = 42
N_SPLITS = 7  # Increased from 5
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

# ========== Enhanced Feature Engineering ==========
def create_features(df):
    # Basic features
    if 'height(cm)' in df.columns and 'weight(kg)' in df.columns:
        df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
        df["BMI_Category"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=[1, 2, 3, 4, 5, 6]
        ).astype(float)  # FIXED: Convert category to float
        df["Weight_to_Height"] = df["weight(kg)"] / df["height(cm)"]
    
    # Interaction features
    if 'age' in df.columns and 'height(cm)' in df.columns:
        df["Age_Height_Interaction"] = df["age"] * df["height(cm)"]
    
    if 'age' in df.columns and 'weight(kg)' in df.columns:
        df["Age_Weight_Interaction"] = df["age"] * df["weight(kg)"]
    
    # Polynomial features
    if 'age' in df.columns:
        df["Age_Squared"] = df["age"] ** 2
    
    if 'height(cm)' in df.columns:
        df["Height_Squared"] = df["height(cm)"] ** 2
    
    if 'weight(kg)' in df.columns:
        df["Weight_Squared"] = df["weight(kg)"] ** 2
    
    # Binning continuous variables
    if 'age' in df.columns:
        df["Age_Binned"] = pd.cut(
            df["age"],
            bins=5,
            labels=False
        ).astype(float)  # FIXED: Make sure it's numeric

    return df


X = create_features(X)
X_test = create_features(X_test)
# Validate feature types and NaNs
print(X.dtypes)  # should be float or int only
assert not X.isnull().any().any(), "X has NaNs!"
assert not X_test.isnull().any().any(), "X_test has NaNs!"

# ========== Advanced Preprocessing ==========
# Power transform for skewed features
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
transformer = PowerTransformer(method='yeo-johnson', standardize=True)
X[numeric_cols] = transformer.fit_transform(X[numeric_cols])
X_test[numeric_cols] = transformer.transform(X_test[numeric_cols])

# Feature selection
selector = SelectKBest(f_classif, k='all')
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
X_test = X_test[selected_features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# ========== Enhanced Optuna Tuning ==========
def tune_lgbm():
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": SEED,
            "n_estimators": 20000,
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 1e2, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs = []
        
        for train_idx, valid_idx in cv.split(X_scaled, y):
            X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            model = LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                callbacks=[
                    early_stopping(200, verbose=False),
                    log_evaluation(0)
                ]
            )
            preds = model.predict_proba(X_valid)[:, 1]
            aucs.append(roc_auc_score(y_valid, preds))
            
            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(aucs)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=HyperbandPruner()
    )
    study.optimize(objective, n_trials=250, timeout=3600)
    print("Best LightGBM trial:", study.best_trial)
    return study.best_trial.params

best_lgbm_params = tune_lgbm()
best_lgbm_params.update({
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": SEED,
    "n_estimators": 20000
})

# ========== Enhanced Model Training ==========
# Prepare meta-feature containers with 5 models instead of 3
oof_preds = np.zeros((len(X), 5))
X_test_preds = np.zeros((len(X_test), 5))

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 1. Tuned LightGBM
    lgbm = LGBMClassifier(**best_lgbm_params)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[early_stopping(200), log_evaluation(0)]
    )
    oof_preds[valid_idx, 0] = lgbm.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 0] += lgbm.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    # 2. Enhanced XGBoost with more tuning
    xgb = XGBClassifier(
    n_estimators=5000,
    learning_rate=0.008,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=SEED,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    enable_categorical=False
)

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=0  # 👈 Now safe
    )

  
    oof_preds[valid_idx, 1] = xgb.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 1] += xgb.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    # 3. Enhanced CatBoost
    cat = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        border_count=128,
        eval_metric="AUC",
        verbose=0,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=200
    )
    cat.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    oof_preds[valid_idx, 2] = cat.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 2] += cat.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    # 4. Logistic Regression with ElasticNet
    logreg = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=0.1,
        l1_ratio=0.5,
        max_iter=1000,
        random_state=SEED,
        class_weight='balanced'
    )
    logreg.fit(X_train, y_train)
    oof_preds[valid_idx, 3] = logreg.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 3] += logreg.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    # 5. Another LightGBM with different params
    lgbm2 = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves=100,
        max_depth=8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight
    )
    lgbm2.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[early_stopping(200), log_evaluation(0)]
    )
    oof_preds[valid_idx, 4] = lgbm2.predict_proba(X_valid)[:, 1]
    X_test_preds[:, 4] += lgbm2.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

# ========== Enhanced Stacking ==========
print("\nTraining meta-model...")

# Option 1: LightGBM as meta-model
meta_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=5,
    random_state=SEED
)
meta_model.fit(oof_preds, y)
meta_preds = meta_model.predict_proba(X_test_preds)[:, 1]

# Option 2: Logistic Regression as meta-model (sometimes works better)
# from sklearn.linear_model import LogisticRegressionCV
# meta_model = LogisticRegressionCV(
#     Cs=10,
#     cv=5,
#     penalty='l2',
#     solver='lbfgs',
#     max_iter=1000,
#     random_state=SEED,
#     class_weight='balanced'
# )
# meta_model.fit(oof_preds, y)
# meta_preds = meta_model.predict_proba(X_test_preds)[:, 1]

# Option 3: Weighted average (simple but effective)
# fold_scores = [roc_auc_score(y, oof_preds[:, i]) for i in range(oof_preds.shape[1])]
# weights = np.array(fold_scores) / sum(fold_scores)
# meta_preds = np.dot(X_test_preds, weights)

# Save submission
submission_df["smoking"] = meta_preds
submission_df = submission_df[["id", "smoking"]]
submission_df.to_csv("submission.csv", index=False)
print("\n✅ Enhanced submission file created successfully!")