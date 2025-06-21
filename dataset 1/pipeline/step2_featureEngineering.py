import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Load original data
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]
X_test = test_df.drop(columns=["id"])

# --- Feature Engineering ---
for df in [X, X_test]:
    df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["Age_BMI"] = df["age"] * df["BMI"]
    df["Height_Weight_Interaction"] = df["height(cm)"] * df["weight(kg)"]
    df["age_bin"] = pd.cut(df["age"], bins=5, labels=False)

# --- Apply PowerTransformer and StandardScaler using training fit ---
pt = PowerTransformer()
X_transformed = pd.DataFrame(pt.fit_transform(X), columns=X.columns)
X_test_transformed = pd.DataFrame(pt.transform(X_test), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_transformed), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_transformed), columns=X.columns)

# Save for model use
X_scaled.to_csv("X_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)

print("✅ X_scaled.csv and X_test_scaled.csv are saved correctly with matching transformations.")
