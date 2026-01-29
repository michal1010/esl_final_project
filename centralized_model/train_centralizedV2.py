from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 42
N_CLIENTS = 5
TEST_SIZE = 0.2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "cox-violent-parsed_filt.csv"

OUT_DIR = REPO_ROOT / "centralized_model" / "centralized_out_like_adithya_raw"

TARGET = "is_recid"
UNIQUE_LABELS = [0, 1]

NUMERIC_FEATURES = [
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "decile_score",
    "jail_time",
]

CATEGORICAL_FEATURES = {
    "sex": ["Male", "Female"],
    "race": ["Other", "Caucasian", "African-American", "Hispanic", "Asian", "Native American"],
}

EXPECTED_DUMMY_COLS: List[str] = []
for col, categories in CATEGORICAL_FEATURES.items():
    for cat in categories:
        EXPECTED_DUMMY_COLS.append(f"{col}_{cat}")


def create_logreg_model() -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        max_iter=1000,
        warm_start=True,
        fit_intercept=True,
    )


def compute_jail_time_days(df: pd.DataFrame) -> pd.Series:
    """Compute jail_time (days) from c_jail_in/c_jail_out if jail_time is missing."""
    if "c_jail_in" not in df.columns or "c_jail_out" not in df.columns:
        return pd.Series([np.nan] * len(df))

    jin = pd.to_datetime(df["c_jail_in"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    jout = pd.to_datetime(df["c_jail_out"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    return (jout - jin).dt.total_seconds() / 86400.0


def preprocess_to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Builds X,y exactly like FL:
    - filters out invalid labels (TARGET != -1)
    - uses NUMERIC_FEATURES + ['sex','race']
    - one-hot encodes sex/race with fixed expected columns/order
    Returns: X (np), y (np), df_used (for race labels etc.)
    """
    df = df.copy()

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column '{TARGET}' in source dataset.")
    for col in ["sex", "race"]:
        if col not in df.columns:
            raise ValueError(f"Missing required categorical column '{col}' in source dataset.")

    if "jail_time" not in df.columns:
        df["jail_time"] = compute_jail_time_days(df)

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing numeric feature '{col}' in source dataset.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[TARGET] != -1].reset_index(drop=True)

    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()

    X = df[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()

    for col in NUMERIC_FEATURES:
        med = float(np.nanmedian(X[col].to_numpy())) if np.isfinite(np.nanmedian(X[col].to_numpy())) else 0.0
        X[col] = X[col].fillna(med)

    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES.keys(), drop_first=False)

    for col in EXPECTED_DUMMY_COLS:
        if col not in X.columns:
            X[col] = 0

    X = X[NUMERIC_FEATURES + EXPECTED_DUMMY_COLS]

    return X.to_numpy(dtype=float), y, df


def calculate_metrics_csv(y_true: np.ndarray, race_dict: Dict[str, List[int]], y_proba: np.ndarray, out_path: Path) -> None:
    """Same structure as Adithya's calcualte_metrics: TPR/FPR per race across thresholds."""
    race_metrics = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        y_pred = (y_proba[:, 1] >= threshold).astype(int)

        for race, indices in race_dict.items():
            tp = fp = fn = tn = 0
            for i in indices:
                if y_true[i] == 1 and y_pred[i] == 1:
                    tp += 1
                elif y_true[i] == 0 and y_pred[i] == 1:
                    fp += 1
                elif y_true[i] == 1 and y_pred[i] == 0:
                    fn += 1
                else:
                    tn += 1

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            race_metrics.append(
                {"race": race, "num_examples": len(indices), "TPR": tpr, "FPR": fpr, "threshold": float(threshold)}
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=race_metrics[-1].keys())
        writer.writeheader()
        writer.writerows(race_metrics)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find source dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X_all, y_all, df_used = preprocess_to_matrix(df)

    X_train_full, X_test, y_train_full, y_test, df_train, df_test = train_test_split(
        X_all,
        y_all,
        df_used,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all,
    )

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X_train_full))
    rng.shuffle(idx)
    client_splits = np.array_split(idx, N_CLIENTS)

    X_train = X_train_full
    y_train = y_train_full

    model = create_logreg_model()
    model.fit(X_train, y_train)

    joblib.dump(model, OUT_DIR / "logreg_model.pkl")

    y_proba = model.predict_proba(X_test)
    loss = float(log_loss(y_test, y_proba, labels=UNIQUE_LABELS))
    acc = float(model.score(X_test, y_test))

    probs1 = y_proba[:, 1]
    y_pred_05 = (probs1 >= 0.5).astype(int)

    overall = {
        "client_style_accuracy": acc,
        "client_style_log_loss": loss,
        "accuracy@0.5": float(accuracy_score(y_test, y_pred_05)),
        "f1@0.5": float(f1_score(y_test, y_pred_05)),
        "roc_auc": float(roc_auc_score(y_test, probs1)) if len(np.unique(y_test)) > 1 else float("nan"),
        "threshold": 0.5,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    df_test = df_test.reset_index(drop=True)
    race_dict: Dict[str, List[int]] = {race: [] for race in CATEGORICAL_FEATURES["race"]}
    for i, race_val in enumerate(df_test["race"].astype(str).tolist()):
        if race_val in race_dict:
            race_dict[race_val].append(i)

    calculate_metrics_csv(y_test, race_dict, y_proba, OUT_DIR / "metrics.csv")

    A = (df_test["race"].astype(str).str.strip() == "African-American").astype(int).to_numpy()

    rows = []
    for aval, gname in [(0, "NonAfricanAmerican"), (1, "AfricanAmerican")]:
        m = (A == aval)
        yt = y_test[m]
        yp = y_pred_05[m]
        n = int(m.sum())
        tp, fp, tn, fn = confusion_counts(yt, yp)
        rows.append(
            {
                "group": gname,
                "A_value": aval,
                "n": n,
                "base_rate_PY1": float(yt.mean()) if n else float("nan"),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "TPR": safe_div(tp, tp + fn),
                "FPR": safe_div(fp, fp + tn),
                "SelectionRate": float(yp.mean()) if n else float("nan"),
                "Accuracy": float((yt == yp).mean()) if n else float("nan"),
            }
        )
    gm = pd.DataFrame(rows)
    gm.to_csv(OUT_DIR / "group_metrics_binary.csv", index=False)

    if len(gm) == 2:
        g0 = gm[gm["A_value"] == 0].iloc[0]
        g1 = gm[gm["A_value"] == 1].iloc[0]
        fairness = {
            "demographic_parity_diff_A1_minus_A0": float(g1["SelectionRate"] - g0["SelectionRate"]),
            "equal_opportunity_diff_TPR_A1_minus_A0": float(g1["TPR"] - g0["TPR"]),
            "fpr_diff_A1_minus_A0": float(g1["FPR"] - g0["FPR"]),
            "equalized_odds_diff_max_abs_tpr_fpr": float(max(abs(g1["TPR"] - g0["TPR"]), abs(g1["FPR"] - g0["FPR"]))),
        }
    else:
        fairness = {}

    metrics_out = {
        "source_data": str(DATA_PATH),
        "split_recreation": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "n_clients": N_CLIENTS,
            "note": "This recreates the random 5-client split + test set from the raw source dataset. "
                    "If your existing federated CSVs were created with different stratification/seed, results will differ slightly.",
        },
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "dummy_cols": EXPECTED_DUMMY_COLS},
        "model": {"type": "LogisticRegression", "solver": "saga", "max_iter": 1000, "warm_start": True, "fit_intercept": True},
        "overall": overall,
        "fairness_binary_AA_vs_nonAA@0.5": fairness,
        "client_partition_sizes": [int(len(s)) for s in client_splits],
        "outputs": ["metrics.csv", "group_metrics_binary.csv", "metrics.json", "logreg_model.pkl"],
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Saved outputs to:", OUT_DIR)
    print("Overall:", overall)
    print("Fairness (AA vs non-AA) @0.5:", fairness)


if __name__ == "__main__":
    main()
