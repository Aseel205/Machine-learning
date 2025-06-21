import os, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score  # במקום accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


DATASETS_DIR = "datasets"
TEST_DIR     = "test"
SEED         = 42
N_ITER = 12
CV_FOLDS     = 4    

def _search(name, model, params, X, y):
    t0 = time.time()
    print(f"\n>>> Starting RandomizedSearchCV for model: {name}")
    cv = RandomizedSearchCV(
        model,
        params,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="balanced_accuracy",  # ✅ שינוי קריטי
        n_jobs=-1,
        random_state=SEED,
        verbose=3
    )
    cv.fit(X, y)
    elapsed = time.time() - t0
    return dict(
        model=name,
        best_score=cv.best_score_,
        best_params=cv.best_params_,
        elapsed=elapsed,
        estimator=cv.best_estimator_
    )

def train_predict(X_train, y_train, X_test):
    grids = {
            "KNN": (
                KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 7]}
            ),

        "Logistic": (
            LogisticRegression(
                solver="saga", max_iter=7000, n_jobs=-1, class_weight="balanced"
            ),
            {"C": np.logspace(-3, 2, 7)}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced"),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
        ),
        "GradientBoost": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=SEED),
            {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}
        ),
        "XGBoost": (
            XGBClassifier(tree_method="hist", predictor="cpu_predictor",
                          random_state=SEED, n_jobs=-1, verbosity=0),
            {"n_estimators": [150, 300], "max_depth": [3, 5, 7],
             "learning_rate": [0.05, 0.1]}
        ),
    }

    results = Parallel(n_jobs=len(grids))(
        delayed(_search)(name, est, p, X_train, y_train)
        for name, (est, p) in grids.items()
    )
    df = (
        pd.DataFrame(results)
        .sort_values("best_score", ascending=False)
        .reset_index(drop=True)
    )
    best_estimator = df.loc[0, "estimator"]
    best_estimator.fit(X_train, y_train)

    # דיווח מסודר למסך
    print("\n===== Model Summary (Balanced Accuracy) =====")
    print(df.drop(columns="estimator").to_string(index=False, formatters={
        "best_score": "{:.4f}".format,
        "elapsed":    "{:.1f}s".format
    }))

    # שמירת דוח
    os.makedirs(TEST_DIR, exist_ok=True)
    df.drop(columns="estimator").to_csv(
        os.path.join(TEST_DIR, "model_report.csv"), index=False
    )

    return best_estimator.predict(X_test)



##### DON'T TOUCH FROM THE LECTURER!


if __name__ == '__main__':
    # --- Configuration ---
    # Enter your FULL 9-digit student ID here (e.g., '123456789')
    student_id_full = '213758758'  # CHANGE THIS TO YOUR FULL STUDENT ID ###

    # We use the LAST 5 digits for file naming convention
    student_id = student_id_full[-5:]

    # --- Sanity Check ---
    if len(student_id_full) != 9 or not student_id_full.isdigit():
        print(
            f"Error: Entered student ID '{student_id_full}' is not a valid 9-digit number.")
        print("Please enter your complete 9-digit ID.")
        exit()
    print(
        f"Using derived ID for files: {student_id} (from full ID: {student_id_full})")

    # List of available IDs can be found in definitions.py (uses 5-digit format) --> Removed as students don't need definitions.py

    # --- Load Data ---
    # Assumes student data file is in the 'datasets' directory relative to this script
    train_file = os.path.join(DATASETS_DIR, f'{student_id}_train.npz')
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        print(f"Contact your TA")
        exit()

    data = np.load(train_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']

    print(f'Running for student ID: {student_id}')
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')

    start_time = datetime.now()
    print(f"\n🟢 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Train and Predict ---
    test_predictions = train_predict(X_train, y_train, X_test)

    # --- Save Predictions ---
    # Ensure the output directory exists
    os.makedirs(TEST_DIR, exist_ok=True)

    pred_file = os.path.join(TEST_DIR, f'{student_id}_test_predictions.npz')
    np.savez(pred_file, test_predictions=test_predictions)

    print(f'\nPredictions saved to {pred_file}')
    print('Submission file created successfully!')
    end_time = datetime.now()
    print(f"✅ Training finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱ Total duration: {str(end_time - start_time)}")
