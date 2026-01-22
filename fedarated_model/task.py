
import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Dataset paths and initialization
DATA_PATH = "federated_data_random"
CLIENTS = [
    "client_random_1.csv",
    "client_random_2.csv",
    "client_random_3.csv",
    "client_random_4.csv",
    "client_random_5.csv",
]

FEATURES = [
    "sex", "age", "race",
    "priors_count", "juv_fel_count",
    "juv_misd_count", "juv_other_count","decile_score"
    ]

TARGET = "is_recid"
UNIQUE_LABELS = [0, 1]

# Logistic Regression initialization
def create_logreg_model():
    return LogisticRegression(
        solver="saga",
        max_iter=100,        
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
def load_data_by_cid(cid: int):
    df = pd.read_csv(os.path.join(DATA_PATH, CLIENTS[cid]))
    df = df[df[TARGET] != -1]
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Encode categorical inputs
    X["sex"] = pd.Categorical(
        X["sex"],
        ["Male", "Female"],
    ).codes

    X["race"] = pd.Categorical(
        X["race"],
        [
            "Other",
            "Caucasian",
            "African-American",
            "Hispanic",
            "Asian",
            "Native American",
        ],
    ).codes

    split = int(0.8 * len(X))
    X_train, X_test = X[:split].values, X[split:].values
    y_train, y_test = y[:split].values, y[split:].values

    return X_train, y_train, X_test, y_test
