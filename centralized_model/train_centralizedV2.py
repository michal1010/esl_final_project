from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

SEED = 42
THRESHOLD = 0.5
N_CLIENTS = 5

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
for col, categories in CATEGORICAL_FEATURES.items():
    for cat in categories:
        EXPECTED_DUMMY_COLS.append(f"{col}_{cat}")


def create_logreg_model() -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        max_iter=5000,
        tol=1e-3,
        warm_start=True,
        fit_intercept=True,
        random_state=SEED,
    )


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = df.copy()

    if TARGET not in df.columns:
        raise ValueError(f"Missing target '{TARGET}' in {DATA_PATH.name}. Found: {list(df.columns)}")

    df = df[df[TARGET] != -1].reset_index(drop=True)

    required = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys()) + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{DATA_PATH.name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()

    X = df[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            med = float(np.nanmedian(X[col].to_numpy()))
            if not np.isfinite(med):
                med = 0.0
            X[col] = X[col].fillna(med)

    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES.keys(), drop_first=False)

    for col in EXPECTED_DUMMY_COLS:
        if col not in X.columns:
            X[col] = 0

    X = X[NUMERIC_FEATURES + EXPECTED_DUMMY_COLS]

    return X.to_numpy(dtype=float), y, df


def threshold_metrics_by_race(y_true: np.ndarray, df_used: pd.DataFrame, y_proba: np.ndarray, out_path: Path) -> None:
    race_dict: Dict[str, List[int]] = {r: [] for r in CATEGORICAL_FEATURES["race"]}
    for i, rv in enumerate(df_used["race"].astype(str).tolist()):
        rv = rv.strip()
        if rv in race_dict:
            race_dict[rv].append(i)

    rows = []
    for thr in np.arange(0.1, 1.0, 0.1):
        y_pred = (y_proba[:, 1] >= thr).astype(int)

        for race, idxs in race_dict.items():
            tp = fp = fn = tn = 0
            for j in idxs:
                if y_true[j] == 1 and y_pred[j] == 1:
                    tp += 1
                elif y_true[j] == 0 and y_pred[j] == 1:
                    fp += 1
                elif y_true[j] == 1 and y_pred[j] == 0:
                    fn += 1
                else:
                    tn += 1

            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0

            rows.append({"race": race, "num_examples": len(idxs), "TPR": tpr, "FPR": fpr, "threshold": float(thr)})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["race", "num_examples", "TPR", "FPR", "threshold"])
        writer.writeheader()
        writer.writerows(rows)


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

    df_raw = pd.read_csv(DATA_PATH)

    df_raw = df_raw.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_all, y_all, df_used = preprocess(df_raw)

    indices = np.arange(len(y_all))
    parts = np.array_split(indices, N_CLIENTS + 1)

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

    threshold_metrics_by_race(y_test, df_test, y_proba, OUT_DIR / "metrics.csv")

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
        }
    else:
        fairness = {}

    metrics_out = {
        "data": str(DATA_PATH),
        "model": {
            "type": "LogisticRegression",
            "solver": "saga",
            "max_iter": 5000,
            "tol": 1e-3,
            "warm_start": True,
            "fit_intercept": True,
            "random_state": SEED,
        },
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "dummy_cols": EXPECTED_DUMMY_COLS},
        "overall": overall,
        "fairness_binary_AA_vs_nonAA@0.5": fairness,
        "artifacts": [
            "logreg_model.pkl",
            "split_indices_federated_style.npz",
            "metrics.csv",
            "group_metrics_binary.csv",
            "metrics.json",
        ],
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Saved outputs to:", OUT_DIR)
    print("Overall:", overall)
    print("Fairness:", fairness)


if __name__ == "__main__":
    main()
