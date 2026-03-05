import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(123)

trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv')

categorical_cols = ['cut', 'color', 'clarity']

trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)

X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
model = LinearRegression()
model.fit(X_trn, y_trn)

yhat_lm = model.predict(X_tst)

out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_K23115695.csv', index=False)

if os.path.exists("CW1_test_with_true_outcome.csv"):
    tst = pd.read_csv("CW1_test_with_true_outcome.csv")
    y_tst = tst["outcome"].values

    def r2_fn(yhat):
        eps = y_tst - yhat
        rss = np.sum(eps ** 2)
        tss = np.sum((y_tst - y_tst.mean()) ** 2)
        return 1 - (rss / tss)

    print("Local R2:", r2_fn(yhat_lm))
else:
    print("Saved submission file. True outcomes not available locally.")

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "CW1_train.csv"
TEST_PATH = ROOT / "CW1_test.csv"

K_NUMBER = "K23115695"
SUBMISSION_PATH = ROOT / f"CW1_submission_{K_NUMBER}.csv"

trn = pd.read_csv(TRAIN_PATH)
X_tst = pd.read_csv(TEST_PATH)

y = trn["outcome"]
X = trn.drop(columns=["outcome"])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
)

base_model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", base_model),
])

param_dist = {
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__max_leaf_nodes": [15, 31, 63, 127],
    "model__min_samples_leaf": [5, 10, 20, 50],
    "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
    "model__max_depth": [None, 3, 5, 7],
    "model__max_iter": [300, 600, 1000],
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scorer = make_scorer(r2_score)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=30,
    scoring=scorer,
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

search.fit(X, y)

print("Best CV R2:", search.best_score_)
print("Best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

best_pipe = search.best_estimator_

best_pipe.fit(X, y)
yhat = best_pipe.predict(X_tst)

out = pd.DataFrame({"yhat": yhat})
out.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission: {SUBMISSION_PATH}")

TRUE_TEST_PATH = ROOT / "CW1_test_with_true_outcome.csv"

if os.path.exists(TRUE_TEST_PATH):
    tst = pd.read_csv(TRUE_TEST_PATH)
    y_tst = tst["outcome"].values

    def r2_fn(yhat_local):
        eps = y_tst - yhat_local
        rss = np.sum(eps ** 2)
        tss = np.sum((y_tst - y_tst.mean()) ** 2)
        return 1 - (rss / tss)

    print("Local R2 (marker-only file):", r2_fn(yhat))
else:
    print("True outcomes not available locally (expected).")