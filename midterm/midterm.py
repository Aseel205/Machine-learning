import os, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score  # 🔁 שונה מ-accuracy_score

DATASETS_DIR = "datasets"
TEST_DIR = "test"
SEED = 42

# ----------------- XGBoost Parameters -----------------
clf = XGBClassifier(
    tree_method="hist",
    predictor="cpu_predictor",
    random_state=SEED,
    n_jobs=-1,
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
# -----------------------------------------------------

def train_predict(X_train, y_train, X_test):
    print(f"\n>>> Using model: XGBoost")

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,
        random_state=SEED,
        stratify=y_train
    )

    clf.fit(X_sub, y_sub)

    y_val_pred = clf.predict(X_val)
    val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)  # 🔁
    print(f"Validation balanced accuracy: {val_bal_acc:.4f}")

    clf.fit(X_train, y_train)

    os.makedirs(TEST_DIR, exist_ok=True)
    pd.DataFrame([{
        "model": "XGBoost",
        "val_balanced_accuracy": val_bal_acc,  # 🔁
        "params": clf.get_params()
    }]).to_csv(
        os.path.join(TEST_DIR, "single_model_report.csv"),
        index=False
    )

    return clf.predict(X_test)


if __name__ == '__main__':
    student_id_full = '213758758'
    student_id = student_id_full[-5:]

    if len(student_id_full) != 9 or not student_id_full.isdigit():
        print(f"Error: Entered student ID '{student_id_full}' is not a valid 9-digit number.")
        exit()

    print(f"Using derived ID for files: {student_id} (from full ID: {student_id_full})")

    train_file = os.path.join(DATASETS_DIR, f'{student_id}_train.npz')
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        exit()

    data = np.load(train_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')

    test_predictions = train_predict(X_train, y_train, X_test)

    os.makedirs(TEST_DIR, exist_ok=True)
    pred_file = os.path.join(TEST_DIR, f'{student_id}_test_predictions.npz')
    np.savez(pred_file, test_predictions=test_predictions)

    print(f'\nPredictions saved to {pred_file}')
    print('Submission file created successfully!')
