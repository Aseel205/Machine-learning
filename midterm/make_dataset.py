import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
from collections import Counter

# === CONFIG ===
n_samples = 18000       # ✅ 18,000 דגימות
n_features = 180        # ✅ 180 תכונות
n_informative = 120     # אפשר לשנות לפי צורך
n_classes = 3
student_id = "213758758"
filename = f"datasets/{student_id[-5:]}_train.npz"

# === DATA GENERATION ===
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_features - n_informative,
    n_classes=n_classes,
    weights=[0.4, 0.4, 0.2],  # איזון חלקי
    flip_y=0.01,              # מעט רעש
    class_sep=1.0,            # מידת הפרדה בין המחלקות
    random_state=42
)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === SAVE ===
os.makedirs('datasets', exist_ok=True)
np.savez(filename, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# === INFO ===
print(f"✅ Dataset saved to '{filename}'")
print(f"🔢 X_train shape: {X_train.shape}")
print(f"🔢 X_test shape: {X_test.shape}")
print(f"🔤 Class distribution: {Counter(y)}")
