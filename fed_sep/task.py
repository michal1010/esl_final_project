
import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from sklearn.metrics import log_loss
import csv
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

# This reweights the loss functions to make the losses scaled uniformly across races for a specific label 
def compute_independence_weights(y, sensitive_attr):
    """
    y: labels (n,)
    sensitive_attr: group attribute (n,)
    """
    df = pd.DataFrame({
        "y": y,
        "a": sensitive_attr
    })

    # P(Y=y)
    py = df["y"].value_counts(normalize=True)

    # P(Y=y | A=a)
    pay = df.groupby("a")["y"].value_counts(normalize=True)

    weights = []
    for _, row in df.iterrows():
        a = row["a"]
        y_val = row["y"]
        w = py[y_val] / pay[a][y_val]
        weights.append(w)

    return np.array(weights)

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
def load_data_by_cid(cid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray]:
    df = pd.read_csv(os.path.join(DATA_PATH, CLIENTS[cid]))
    df = df[df[TARGET] != -1]  # remove invalid labels
    
    df2 = pd.read_csv(os.path.join(DATA_PATH,TEST))
    df2=df2[df2[TARGET]!= -1]
    # Split features and target
    X = df[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()
    X_test=df2[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()
    y_test=df2[TARGET].copy()
    y = df[TARGET].copy()
    race = df["race"].values[:len(y)]
    sample_weights = compute_independence_weights(y, race)
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

    return X_train.values, y_train.values, X_test.values, y_test.values,sample_weights


def calculate_metrics(y_test, race_dict, y_pred):

    race_metrics = []

    for race, indices in race_dict.items():
        if len(indices) == 0:
            race_metrics.append({
                'race': race,
                "num_examples": 0,
                "TPR": None,
                "FPR": None
            })
            continue

        tp = fp = fn = tn = 0

        for i in indices:
            if y_test[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_test[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_test[i] == 1 and y_pred[i] == 0:
                fn += 1
            else:
                tn += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        race_metrics.append({
            'race': race,
            "num_examples": len(indices),
            "TPR": tpr,
            "FPR": fpr
        })


    with open(f'metrics.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=race_metrics[-1].keys())
        writer.writeheader()
        writer.writerows(race_metrics)

def calculate_roc_metrics(y_test, race_dict, y_proba):
    race_metrics = []

    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_proba >= threshold).astype(int)

        for race, indices in race_dict.items():
            tp = fp = fn = tn = 0

            for i in indices:
                if y_test[i] == 1 and y_pred[i] == 1:
                    tp += 1
                elif y_test[i] == 0 and y_pred[i] == 1:
                    fp += 1
                elif y_test[i] == 1 and y_pred[i] == 0:
                    fn += 1
                else:
                    tn += 1

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            race_metrics.append({
                'race': race,
                "num_examples": len(indices),
                "TPR": tpr,
                "FPR": fpr,
                'threshold': threshold
            })

    with open(f'metrics_roc.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=race_metrics[-1].keys())
        writer.writeheader()
        writer.writerows(race_metrics)

def calculate_PPV_NPV(y_test, race_dict, y_pred):
    race_metrics = []

    for race, indices in race_dict.items():
        if len(indices) == 0:
            race_metrics.append({
                'race': race,
                "num_examples": 0,
                "TPR": None,
                "FPR": None
            })
            continue

        tp = fp = fn = tn = 0

        for i in indices:
            if y_test[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_test[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_test[i] == 1 and y_pred[i] == 0:
                fn += 1
            else:
                tn += 1

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (fn + tn) if (fn + tn) > 0 else 0.0

        race_metrics.append({
            'race': race,
            "num_examples": len(indices),
            "PPV": ppv,
            "NPV": npv
        })

        with open(f'metrics_ppv.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=race_metrics[-1].keys())
            writer.writeheader()
            writer.writerows(race_metrics)