from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TEST_SIZE = 0.2

LABEL_COL = "is_recid"
SENSITIVE_COL = "race"
SENSITIVE_POS_VALUE = "African-American"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "cox-violent-parsed_filt.csv"

OUT_DIR = REPO_ROOT / "centralized_model" / "centralized_out"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> pd.DataFrame:
    rows = []
    for aval, gname in [(0, "NonAfricanAmerican"), (1, "AfricanAmerican")]:
        mask = (A == aval)
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(mask.sum())
        if n == 0:
            continue

        counts = confusion_counts(yt, yp)
        tp, fp, tn, fn = counts["TP"], counts["FP"], counts["TN"], counts["FN"]

        base_rate = float(yt.mean())
        sel_rate = float(yp.mean())

        tpr = safe_div(tp, tp + fn)
        fpr = safe_div(fp, fp + tn)
        prec = safe_div(tp, tp + fp)
        acc = float((yt == yp).mean())

        rows.append(
            {
                "group": gname,
                "A_value": aval,
                "n": n,
                "base_rate_PY1": base_rate,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "TPR": tpr,
                "FPR": fpr,
                "Precision": prec,
                "Accuracy": acc,
                "SelectionRate": sel_rate,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    for col in [LABEL_COL, SENSITIVE_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in {DATA_PATH.name}. "
                f"Found columns like: {list(df.columns)[:30]} ..."
            )

    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()

    A = (df[SENSITIVE_COL].astype(str).str.strip() == SENSITIVE_POS_VALUE).astype(int).to_numpy()
    drop_cols = {
        LABEL_COL,
        "id", "name", "first", "last",
        "dob", "c_jail_in", "c_jail_out", "r_offense_date", "vr_offense_date", "r_jail_in", "screening_date",
        "c_charge_desc", "r_charge_desc", "vr_charge_desc", "score_text", "v_score_text",
        "decile_score.1", "priors_count.1",
        "event",
    }
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    categorical_cols = [c for c in X_df.columns if X_df[c].dtype == object]
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=SEED,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X_df, y, A, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    overall = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan"),
        "threshold": 0.5,
    }

    gm = group_metrics(y_test, y_pred, A_test)

    if len(gm) == 2:
        g0 = gm[gm["A_value"] == 0].iloc[0]
        g1 = gm[gm["A_value"] == 1].iloc[0]
        dp_diff = float(g1["SelectionRate"] - g0["SelectionRate"])
        tpr_diff = float(g1["TPR"] - g0["TPR"])
        fpr_diff = float(g1["FPR"] - g0["FPR"])
        eq_odds = float(max(abs(tpr_diff), abs(fpr_diff)))
    else:
        dp_diff = tpr_diff = fpr_diff = eq_odds = float("nan")

    fairness = {
        "demographic_parity_diff_A1_minus_A0": dp_diff,
        "equal_opportunity_diff_TPR_A1_minus_A0": tpr_diff,
        "fpr_diff_A1_minus_A0": fpr_diff,
        "equalized_odds_diff_max_abs_tpr_fpr": eq_odds,
        "sensitive_definition": f"A=1 if {SENSITIVE_COL} == '{SENSITIVE_POS_VALUE}'",
    }

    joblib.dump(pipe, OUT_DIR / "sklearn_pipeline.joblib")
    gm.to_csv(OUT_DIR / "group_metrics.csv", index=False)

    metrics_out = {
        "data": str(DATA_PATH),
        "label_col": LABEL_COL,
        "sensitive_col": SENSITIVE_COL,
        "overall": overall,
        "fairness": fairness,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_used": {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        },
        "dropped_cols": sorted([c for c in drop_cols if c in df.columns]),
        "seed": SEED,
        "test_size": TEST_SIZE,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Saved to:", OUT_DIR)
    print("Overall:", overall)
    print("Fairness:", fairness)
    print("Wrote group_metrics.csv and metrics.json")


if __name__ == "__main__":
    main()
