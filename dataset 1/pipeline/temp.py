import pandas as pd

train_labels = pd.read_csv("dataset/train.csv")["smoking"][:15000]
train_labels.to_csv("y.csv", index=False, header=False)
print("✅ y.csv updated with 15000 rows.")
