### 1. Logistic Regression

```python
LogisticRegression(
    solver="saga",       # a fast, stochastic-gradient solver that handles large, sparse data & L1/L2 penalties
    max_iter=7000,       # allow up to 7 000 iterations to converge (prevents ConvergenceWarning)
    n_jobs=-1            # parallelize over CPU cores when possible
),
{"C": np.logspace(-3, 2, 7)}   # inverse regularization strength: [1e-3,1e-2,1e-1,1,10,100,1 000]

suggestions::
LogisticRegression(penalty="elasticnet", l1_ratio=[0.1,0.5,0.9])

```

* **C** smaller → stronger L2 regularization (shrinks coefficients); larger → weaker regularization.
* We sweep across 7 orders of magnitude to find the sweet spot between under- and over-fitting.

---

### 2. Decision Tree

```python
DecisionTreeClassifier(random_state=SEED),
{"max_depth": [None, 5, 10, 20],
 "min_samples_leaf": [1, 5, 10]}
```

* **max\_depth**: maximum depth of the tree

  * `None` = grow until leaves are pure (risk of overfitting),
  * 5/10/20 = progressively simpler trees.
* **min\_samples\_leaf**: minimum number of samples required to remain in a leaf node

  * larger → smoother, more general trees.

---

### 3. Random Forest

```python
RandomForestClassifier(random_state=SEED, n_jobs=-1),
{"n_estimators": [100, 200],
 "max_depth":    [None, 10, 20]}
```

* **n\_estimators**: number of trees in the forest (more → better stability at cost of time).
* **max\_depth**: as above, controls per-tree complexity.

---

### 4. Gradient Boosting

```python
GradientBoostingClassifier(random_state=SEED),
{"n_estimators":   [100, 200],
 "learning_rate":  [0.05, 0.1]}
```

* **n\_estimators**: number of boosting stages (weak trees).
* **learning\_rate**: shrinkage factor (small → slower learning & better generalization, large → faster but risk overfit).

> *Commented-out grid* shows a more aggressive search adding:
>
> * `max_depth` of each tree,
> * `subsample` (row sampling fraction),
> * `colsample_bytree` (feature sampling fraction).

---

### 5. AdaBoost

```python
AdaBoostClassifier(random_state=SEED),
{"n_estimators": [50, 100],
 "learning_rate": [0.5, 1.0]}
```

* **n\_estimators**: number of weak learners (usually small decision stumps).
* **learning\_rate**: weight applied to each classifier’s vote (controls contribution).

---

### 6. XGBoost

```python
XGBClassifier(
    tree_method="hist",         # uses histogram-based split finding (much faster)
    predictor="cpu_predictor",  # pure-CPU prediction
    random_state=SEED,
    n_jobs=-1,
    verbosity=0
),
{"n_estimators":   [150, 300],
 "max_depth":      [3,  5,  7],
 "learning_rate":  [0.05, 0.1]}
```

* **n\_estimators**: boosting rounds.
* **max\_depth**: maximum tree depth.
* **learning\_rate**: as in GradientBoosting.
* `subsample` and `colsample_bytree` are also common XGBoost knobs (not shown here).

---

### Why these settings?

* We pick **small grids** so that `RandomizedSearchCV(n_iter=12)` can explore a diverse mix without exploding runtime.
* We use **3-fold CV** (`cv=CV_FOLDS`) to get a stable accuracy estimate for each hyper-parameter combination.
* **Parallelism** (`n_jobs=-1` & `Parallel(n_jobs=len(grids))`) lets you search all models at once, fully utilizing your CPU.

This setup balances **speed** (to fit within your 1½ hr exam) and **coverage** (to find reasonably good parameter values for each algorithm).
