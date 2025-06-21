import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

# Load OOF predictions
oof_lgb = pd.read_csv("oof_lgb.csv")["lgb_oof"]
oof_xgb = pd.read_csv("oof_xgb.csv")["xgb_oof"]
oof_cat = pd.read_csv("oof_cat.csv")["cat_oof"]

# Load test predictions
test_lgb = pd.read_csv("test_preds_lgb.csv")["lgb_test"]
test_xgb = pd.read_csv("test_preds_xgb.csv")["xgb_test"]
test_cat = pd.read_csv("test_preds_cat.csv")["cat_test"]

# Stack OOF and test predictions
X_meta = pd.concat([oof_lgb, oof_xgb, oof_cat], axis=1)
X_meta.columns = ["lgb", "xgb", "cat"]
X_meta_test = pd.concat([test_lgb, test_xgb, test_cat], axis=1)
X_meta_test.columns = ["lgb", "xgb", "cat"]

# Load target
y = pd.read_csv("dataset/train.csv")["smoking"]

# Train meta-model
meta_model = LogisticRegressionCV(cv=5, scoring="roc_auc", random_state=42, max_iter=1000)
meta_model.fit(X_meta, y)

# Evaluate OOF AUC
oof_meta = meta_model.predict_proba(X_meta)[:, 1]
meta_auc = roc_auc_score(y, oof_meta)
print(f"\n🎯 Meta-model OOF AUC: {meta_auc:.5f}")

# Predict on test set
final_preds = meta_model.predict_proba(X_meta_test)[:, 1]

# Save final prediction
pd.DataFrame({"smoking": final_preds}).to_csv("submission_stacked.csv", index=False)
print("📁 Saved: submission_stacked.csv")
