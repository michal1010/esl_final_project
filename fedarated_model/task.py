
import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
# Dataset paths and initialization
DATA_PATH = "federated_data_random"
CLIENTS = [
    "client_random_1.csv",
    "client_random_2.csv",
    "client_random_3.csv",
    "client_random_4.csv",
    "client_random_5.csv",
]

TEST = "client_random_test.csv"

TARGET = "is_recid"
UNIQUE_LABELS = [0, 1]
FEATURES = [
    "sex", "age", "race",
    "priors_count", "juv_fel_count",
    "juv_misd_count", "juv_other_count","decile_score"
    ]
NUMERIC_FEATURES = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count","decile_score"]
CATEGORICAL_FEATURES = {
    "sex": ["Male", "Female"],
    "race": ["Other", "Caucasian", "African-American", "Hispanic", "Asian", "Native American"],
}

# Precompute the expected dummy columns for all clients
EXPECTED_DUMMY_COLS = []
for col, categories in CATEGORICAL_FEATURES.items():
    # drop_first=True, so skip the first category
    for cat in categories:
        EXPECTED_DUMMY_COLS.append(f"{col}_{cat}")
# Logistic Regression initialization
def create_logreg_model():
    return LogisticRegression(
        solver="saga",
        max_iter=1000,        
        warm_start=True,
        fit_intercept=True,
    )

def set_initial_params(model, n_features: int):
    model.classes_ = np.array(UNIQUE_LABELS)
    model.coef_ = np.zeros((1, n_features))
    model.intercept_ = np.zeros(1)

def get_model_params(model) -> List[np.ndarray]:
    return [model.coef_, model.intercept_]

def set_model_params(model, params: List[np.ndarray]):
    model.coef_ = params[0]
    model.intercept_ = params[1]
    return model

# Data Loading
def load_data_by_cid(cid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(DATA_PATH, CLIENTS[cid]))
    df = df[df[TARGET] != -1]  # remove invalid labels
    df2 = pd.read_csv(os.path.join(DATA_PATH,TEST))
    df2=df2[df2[TARGET]!= -1]
    # Split features and target
    X = df[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()
    X_test=df2[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()
    y_test=df2[TARGET].copy()
    y = df[TARGET].copy()

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES.keys(), drop_first=False)
    X_test=pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES.keys(), drop_first=False)
    # Ensure all dummy columns exist (add missing ones as 0)
    for col in EXPECTED_DUMMY_COLS:
        if col not in X.columns:
            X[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    # Reorder columns to a fixed order
    X = X[NUMERIC_FEATURES + EXPECTED_DUMMY_COLS]
    X_test=X_test[NUMERIC_FEATURES + EXPECTED_DUMMY_COLS]

    # Train/test split
    # split = int(0.8 * len(X))
    X_train = X
    y_train = y

    return X_train, y_train, X_test, y_test
