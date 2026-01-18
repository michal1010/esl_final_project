from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class LogisticModel(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def confusion_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def rates(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    sel = (tp + fp) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    return {"TPR": tpr, "FPR": fpr, "Precision": precision, "Accuracy": acc, "SelectionRate": sel}


@torch.no_grad()
def predict_logits(model: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xb).detach().cpu().numpy()
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts",
        type=str,
        default="/mnt/data/centralized_out",
        help="Directory with model.pt, scaler.joblib, config.json, split_indices.npz",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional override path to CSV (otherwise uses config.json data path)",
    )
    args = ap.parse_args()

    cfg_path = os.path.join(args.artifacts, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path}. Run train_centralized.py first.")

    with open(cfg_path, "r") as f:
        meta = json.load(f)

    data_path = args.data or meta["data"]
    df = pd.read_csv(data_path)

    label_col = meta["label_col"]
    feature_cols = meta["feature_cols"]
    race_cols = meta["race_cols"]

    sens_col = "African_American"
    if sens_col not in df.columns:
        raise ValueError(f"Expected sensitive attribute column '{sens_col}' in data.")

    y = df[label_col].astype(int).to_numpy()

    X_df = df[feature_cols].copy()
    for c in X_df.columns:
        if not np.issubdtype(X_df[c].dtype, np.number):
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    if X_df.isna().any().any():
        X_df = X_df.fillna(0)

    X = X_df.to_numpy().astype(np.float32)

    split_npz = os.path.join(args.artifacts, "split_indices.npz")
    splits = np.load(split_npz)
    test_idx = splits["test_idx"].astype(int)

    scaler = joblib.load(os.path.join(args.artifacts, "scaler.joblib"))
    cont_cols = meta.get("continuous_cols_scaled", [])
    cont_idx = [feature_cols.index(c) for c in cont_cols if c in feature_cols]
    if cont_idx:
        X[:, cont_idx] = scaler.transform(X[:, cont_idx])

    model = LogisticModel(n_features=X.shape[1])
    state = torch.load(os.path.join(args.artifacts, "model.pt"), map_location="cpu")
    model.load_state_dict(state)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    X_test = X[test_idx]
    y_test = y[test_idx]
    A_test = df.loc[test_idx, sens_col].astype(int).to_numpy()

    logits = predict_logits(model, X_test, device)
    prob = sigmoid(logits)
    pred = (prob >= args.threshold).astype(int)

    overall = {
        "threshold": float(args.threshold),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }
    try:
        overall["roc_auc"] = float(roc_auc_score(y_test, prob))
    except Exception:
        overall["roc_auc"] = float("nan")

    rows = []
    for a_val, group_name in [(0, "NonAfricanAmerican"), (1, "AfricanAmerican")]:
        mask = A_test == a_val
        y_g = y_test[mask]
        p_g = prob[mask]
        base_rate = float(np.mean(y_g)) if len(y_g) else float("nan")
        tp, fp, tn, fn = confusion_from_probs(y_g, p_g, args.threshold)
        r = rates(tp, fp, tn, fn)
        rows.append(
            {
                "group": group_name,
                "A_value": int(a_val),
                "n": int(mask.sum()),
                "base_rate_PY1": base_rate,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                **r,
            }
        )

    group_df = pd.DataFrame(rows)

    def get_metric(df_: pd.DataFrame, a_value: int, col: str) -> float:
        v = df_.loc[df_["A_value"] == a_value, col]
        return float(v.iloc[0]) if len(v) else float("nan")

    dp_diff = get_metric(group_df, 1, "SelectionRate") - get_metric(group_df, 0, "SelectionRate")
    tpr_diff = get_metric(group_df, 1, "TPR") - get_metric(group_df, 0, "TPR")
    fpr_diff = get_metric(group_df, 1, "FPR") - get_metric(group_df, 0, "FPR")
    eo_diff = tpr_diff
    eod_diff = float(np.nanmax([abs(tpr_diff), abs(fpr_diff)]))

    fairness = {
        "demographic_parity_diff_A1_minus_A0": float(dp_diff),
        "equal_opportunity_diff_TPR_A1_minus_A0": float(eo_diff),
        "tpr_diff_A1_minus_A0": float(tpr_diff),
        "fpr_diff_A1_minus_A0": float(fpr_diff),
        "equalized_odds_diff_max_abs_tpr_fpr": float(eod_diff),
    }

    out = {
        "data": os.path.abspath(data_path),
        "artifacts": os.path.abspath(args.artifacts),
        "label_col": label_col,
        "sensitive_attribute": sens_col,
        "overall": overall,
        "fairness": fairness,
    }

    os.makedirs(args.artifacts, exist_ok=True)
    with open(os.path.join(args.artifacts, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    group_df.to_csv(os.path.join(args.artifacts, "group_metrics.csv"), index=False)

    print("Wrote:")
    print(" -", os.path.join(args.artifacts, "metrics.json"))
    print(" -", os.path.join(args.artifacts, "group_metrics.csv"))


if __name__ == "__main__":
    main()
