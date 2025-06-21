import pandas as pd

# Load your file (change the filename as needed)
df = pd.read_csv("oof_lgb.csv")  # e.g., 'test_preds_lgb.csv'

# Keep only the 'smoking' column
smoking_only = df["smoking"]

# Save to a new file without the ID
smoking_only.to_csv("oof_lgb.csv", index=False, header=False)  # ready for stacking
