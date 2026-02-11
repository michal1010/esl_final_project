from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, roc_curve

SEED = 42
THRESHOLD = 0.6
N_CLIENTS = 5

SOLVER = "saga"
MAX_ITER = 1000
WARM_START = True
FIT_INTERCEPT = True

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "centralized_dataset.csv"
OUT_DIR = REPO_ROOT / "centralized_model" / "centralized_out"

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
for col, cats in CATEGORICAL_FEATURES.items():
    for c in cats:
        EXPECTED_DUMMY_COLS.append(f"{col}_{c}")


def create_logreg_model() -> LogisticRegression:
    return LogisticRegression(
        solver=SOLVER,
        max_iter=MAX_ITER,
        warm_start=WARM_START,
        fit_intercept=FIT_INTERCEPT,
        random_state=SEED,
    )


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = df.copy()

    if TARGET not in df.columns:
        raise ValueError(f"Missing target '{TARGET}'. Found: {list(df.columns)}")

    df = df[df[TARGET] != -1].reset_index(drop=True)

    required = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys()) + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int).clip(0, 1).to_numpy()
    X = df[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X[NUMERIC_FEATURES].isna().any().any():
        raise ValueError(
            "NaNs found in numeric features. "
            "To match federated pipeline, this script does not impute. Clean the dataset upstream."
        )
    for c in CATEGORICAL_FEATURES.keys():
        X[c] = X[c].astype(str).str.strip()

    X = pd.get_dummies(X, columns=list(CATEGORICAL_FEATURES.keys()), drop_first=False)

    for col in EXPECTED_DUMMY_COLS:
        if col not in X.columns:
            X[col] = 0

    X = X[NUMERIC_FEATURES + EXPECTED_DUMMY_COLS]

    return X.to_numpy(dtype=float), y, df


def save_threshold_sweep_by_race(
    y_true: np.ndarray, df_used: pd.DataFrame, probs1: np.ndarray, out_path: Path, n_thresholds: int = 101
) -> None:
    race_series = df_used["race"].astype(str).str.strip()
    race_dict: Dict[str, np.ndarray] = {r: (race_series == r).to_numpy() for r in CATEGORICAL_FEATURES["race"]}

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []

    for thr in thresholds:
        y_pred = (probs1 >= thr).astype(int)
        for race, mask in race_dict.items():
            yt = y_true[mask]
            yp = y_pred[mask]
            if len(yt) == 0:
                continue
            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            rows.append({"race": race, "num_examples": int(mask.sum()), "TPR": tpr, "FPR": fpr, "threshold": float(thr)})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["race", "num_examples", "TPR", "FPR", "threshold"])
        writer.writeheader()
        writer.writerows(rows)


def save_true_roc_points_by_race(y_true: np.ndarray, df_used: pd.DataFrame, probs1: np.ndarray, out_path: Path) -> None:
    rows = []
    races = df_used["race"].astype(str).str.strip()

    for race in CATEGORICAL_FEATURES["race"]:
        mask = (races == race).to_numpy()
        yt = y_true[mask]
        ps = probs1[mask]
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            continue
        fpr, tpr, thr = roc_curve(yt, ps)
        auc = roc_auc_score(yt, ps)
        for f, t, th in zip(fpr, tpr, thr):
            rows.append({"race": race, "FPR": float(f), "TPR": float(t), "threshold": float(th), "auc": float(auc), "n": int(len(yt))})

    pd.DataFrame(rows).to_csv(out_path, index=False)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
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
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_all, y_all, df_used = preprocess(df)

    idx = np.arange(len(y_all))
    parts = np.array_split(idx, N_CLIENTS + 1)
    test_idx = parts[0]
    train_idx = np.concatenate(parts[1:])

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    df_test = df_used.iloc[test_idx].reset_index(drop=True)

    model = create_logreg_model()
    model.fit(X_train, y_train)

    joblib.dump(model, OUT_DIR / "logreg_model.pkl")
    np.savez(OUT_DIR / "split_indices_federated_style.npz", train_idx=train_idx, test_idx=test_idx)

    y_proba = model.predict_proba(X_test)
    probs1 = y_proba[:, 1]
    y_pred = (probs1 >= THRESHOLD).astype(int)

    overall = {
        "accuracy@0.5": float(accuracy_score(y_test, y_pred)),
        "f1@0.5": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, probs1)) if len(np.unique(y_test)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_test, y_proba, labels=UNIQUE_LABELS)),
        "threshold": THRESHOLD,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_rate_test": float(y_test.mean()),
        "selection_rate@0.5": float(y_pred.mean()),
        "seed": SEED,
        "split_style": f"shuffle(seed={SEED}) + array_split into {N_CLIENTS+1} parts (part0 test, rest train)",
    }

    save_threshold_sweep_by_race(y_test, df_test, probs1, OUT_DIR / "metrics.csv", n_thresholds=101)
    save_true_roc_points_by_race(y_test, df_test, probs1, OUT_DIR / "roc_points.csv")

    A = (df_test["race"].astype(str).str.strip() == "African-American").astype(int).to_numpy()

    rows = []
    for aval, gname in [(0, "NonAfricanAmerican"), (1, "AfricanAmerican")]:
        m = (A == aval)
        yt = y_test[m]
        yp = y_pred[m]
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
            "equalized_odds_diff_max_abs_tpr_fpr": float(
                max(abs(g1["TPR"] - g0["TPR"]), abs(g1["FPR"] - g0["FPR"]))
            ),
            "sensitive_definition": "A=1 if race == 'African-American'",
        }
    else:
        fairness = {}

    metrics_out = {
        "data": str(DATA_PATH),
        "model": {
            "type": "LogisticRegression",
            "solver": SOLVER,
            "max_iter": MAX_ITER,
            "warm_start": WARM_START,
            "fit_intercept": FIT_INTERCEPT,
            "random_state": SEED,
        },
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "dummy_cols": EXPECTED_DUMMY_COLS},
        "overall": overall,
        "fairness_binary_AA_vs_nonAA@0.5": fairness,
        "outputs": [
            "logreg_model.pkl",
            "split_indices_federated_style.npz",
            "metrics.csv",
            "roc_points.csv",
            "group_metrics_binary.csv",
            "metrics.json",
        ],
        "notes": [
            "Centralized uses the federated feature set and preprocessing (no feature selection).",
            "No numeric imputation is performed (clean upstream dataset required).",
        ],
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Saved outputs to:", OUT_DIR)
    print("Overall:", overall)
    print("Fairness:", fairness)


if __name__ == "__main__":
    main()