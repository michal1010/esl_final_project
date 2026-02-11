from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

from fairlearn.postprocessing import ThresholdOptimizer

# Add fed_ind to path to import from task.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fed_ind"))
from task import NUMERIC_FEATURES, CATEGORICAL_FEATURES, EXPECTED_DUMMY_COLS

SEED = 42
BASE_THRESHOLD = 0.5
N_CLIENTS = 5

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "centralized_dataset.csv"
OUT_DIR = REPO_ROOT / "centralized_model" / "centralized_out"

TARGET = "is_recid"


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
    """Preprocess using the same pipeline as fed_ind/task.py"""
    df = df.copy()
    if TARGET not in df.columns:
        raise ValueError(f"Missing target '{TARGET}'")

    df = df[df[TARGET] != -1].reset_index(drop=True)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
    X = df[NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())].copy()

    # Handle numeric features (no imputation like in fed_ind)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            med = np.nanmedian(X[col])
            if not np.isfinite(med):
                med = 0.0
            X[col] = X[col].fillna(med)

    # One-hot encode categorical (same as fed_ind)
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES.keys(), drop_first=False)
    
    # Ensure all expected dummy columns exist
    for col in EXPECTED_DUMMY_COLS:
        if col not in X.columns:
            X[col] = 0
    
    # Reorder to match fed_ind
    X = X[NUMERIC_FEATURES + EXPECTED_DUMMY_COLS]
    
    return X.to_numpy(dtype=float), y, df


def compute_group_table(y_true, y_pred, race_column, threshold_label):
    """Compute comprehensive fairness metrics per group."""
    rows = []
    
    # Get all unique races in the data
    unique_races = sorted(race_column.unique())
    
    for race_name in unique_races:
        mask = race_column == race_name
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(mask.sum())
        
        if n == 0:
            continue

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        # Core metrics
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")  # Specificity
        fnr = fn / (fn + tp) if (fn + tp) else float("nan")
        
        # Additional metrics
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        selection_rate = float(yp.mean()) if n > 0 else float("nan")  # P(Ŷ=1)
        base_rate = float(yt.mean()) if n > 0 else float("nan")  # P(Y=1)
        accuracy = float((yt == yp).mean()) if n > 0 else float("nan")

        rows.append(
            {
                "race": race_name,
                "num_examples": n,
                "base_rate": base_rate,
                "selection_rate": selection_rate,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "TPR": tpr,
                "FPR": fpr,
                "TNR": tnr,
                "FNR": fnr,
                "Precision": precision,
                "Accuracy": accuracy,
                "threshold": threshold_label,
            }
        )
    return pd.DataFrame(rows)


def compute_fairness_metrics(df_group: pd.DataFrame) -> dict:
    """Compute fairness disparity metrics from group table."""
    if len(df_group) < 2:
        return {}
    
    # Compute max disparities across all race pairs
    selection_rates = df_group["selection_rate"].values
    tprs = df_group["TPR"].values
    fprs = df_group["FPR"].values
    precisions = df_group["Precision"].values
    
    # Filter out NaN values before computing differences
    selection_rates_valid = selection_rates[~np.isnan(selection_rates)]
    tprs_valid = tprs[~np.isnan(tprs)]
    fprs_valid = fprs[~np.isnan(fprs)]
    precisions_valid = precisions[~np.isnan(precisions)]
    
    metrics = {
        "demographic_parity_diff_max": float(np.max(selection_rates_valid) - np.min(selection_rates_valid)) if len(selection_rates_valid) > 0 else float("nan"),
        "equal_opportunity_diff_max": float(np.max(tprs_valid) - np.min(tprs_valid)) if len(tprs_valid) > 0 else float("nan"),
        "equalized_odds_tpr_diff_max": float(np.max(tprs_valid) - np.min(tprs_valid)) if len(tprs_valid) > 0 else float("nan"),
        "equalized_odds_fpr_diff_max": float(np.max(fprs_valid) - np.min(fprs_valid)) if len(fprs_valid) > 0 else float("nan"),
        "predictive_parity_diff_max": float(np.max(precisions_valid) - np.min(precisions_valid)) if len(precisions_valid) > 0 else float("nan"),
    }
    
    # Compute equalized odds as max of TPR and FPR differences
    metrics["equalized_odds_max"] = float(max(
        metrics["equalized_odds_tpr_diff_max"] if not np.isnan(metrics["equalized_odds_tpr_diff_max"]) else 0,
        metrics["equalized_odds_fpr_diff_max"] if not np.isnan(metrics["equalized_odds_fpr_diff_max"]) else 0
    ))
    
    # Also compute specific African-American vs rest for comparison
    if "African-American" in df_group["race"].values:
        aa = df_group[df_group["race"] == "African-American"].iloc[0]
        non_aa = df_group[df_group["race"] != "African-American"]
        
        # Average metrics for non-African-American groups
        avg_sel_rate = non_aa["selection_rate"].mean()
        avg_tpr = non_aa["TPR"].mean()
        avg_fpr = non_aa["FPR"].mean()
        avg_prec = non_aa["Precision"].mean()
        
        metrics["demographic_parity_diff_AA_vs_rest"] = float(aa["selection_rate"] - avg_sel_rate)
        metrics["equal_opportunity_diff_AA_vs_rest"] = float(aa["TPR"] - avg_tpr)
        metrics["fpr_diff_AA_vs_rest"] = float(aa["FPR"] - avg_fpr)
        metrics["precision_diff_AA_vs_rest"] = float(aa["Precision"] - avg_prec)
    
    return metrics


def compute_roc_points(y_true, probs, race_column, scenario_label):
    """Compute ROC curve points for each race group."""
    roc_rows = []
    
    # Get all unique races
    unique_races = sorted(race_column.unique())
    
    for race_name in unique_races:
        mask = race_column == race_name
        y_race = y_true[mask]
        prob_race = probs[mask]
        
        if len(y_race) == 0 or len(np.unique(y_race)) < 2:
            # Cannot compute ROC if no samples or only one class present
            continue
        
        fpr, tpr, thresholds = roc_curve(y_race, prob_race)
        roc_auc = auc(fpr, tpr)
        
        # Store all points
        for i in range(len(fpr)):
            roc_rows.append({
                "race": race_name,
                "scenario": scenario_label,
                "FPR": float(fpr[i]),
                "TPR": float(tpr[i]),
                "threshold": float(thresholds[i]) if i < len(thresholds) else 1.0,
                "auc": float(roc_auc),
            })
    
    return pd.DataFrame(roc_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH).sample(frac=1, random_state=SEED).reset_index(drop=True)
    X_all, y_all, df_used = preprocess(df_raw)

    idx = np.arange(len(y_all))
    parts = np.array_split(idx, N_CLIENTS + 1)
    test_idx, train_idx = parts[0], np.concatenate(parts[1:])
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    df_test = df_used.iloc[test_idx].reset_index(drop=True)

    # -----------------------------
    # Train base model
    # -----------------------------
    print("Training base model...")
    base_model = create_logreg_model()
    base_model.fit(X_train, y_train)
    joblib.dump(base_model, OUT_DIR / "logreg_model.pkl")

    probs_test = base_model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # Sensitive attribute - use full race column
    # -----------------------------
    race_test = df_test["race"].astype(str).str.strip()
    
    print(f"\nTest set: {len(X_test)} samples")
    print(f"Race distribution:")
    race_counts = race_test.value_counts()
    for race, count in race_counts.items():
        print(f"  {race}: {count} ({100*count/len(race_test):.1f}%)")
    print(f"  Overall recidivism rate: {100*y_test.mean():.1f}%")

    all_results = {}
    all_roc_points = []

    # =====================================================
    # 1) Baseline (fixed threshold = 0.5)
    # =====================================================
    print("\n" + "="*60)
    print("1. BASELINE (threshold = 0.5)")
    print("="*60)
    
    y_pred_base = (probs_test >= 0.5).astype(int)
    df_baseline = compute_group_table(y_test, y_pred_base, race_test, "0.5")
    df_baseline.to_csv(OUT_DIR / "group_metrics_baseline.csv", index=False)
    
    # Compute ROC for baseline
    roc_baseline = compute_roc_points(y_test, probs_test, race_test, "baseline")
    all_roc_points.append(roc_baseline)
    
    fairness_base = compute_fairness_metrics(df_baseline)
    all_results["baseline"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_base)),
        "f1": float(f1_score(y_test, y_pred_base)),
        "auc": float(roc_auc_score(y_test, probs_test)),
        "fairness": fairness_base,
    }
    
    print(df_baseline.to_string(index=False))
    print(f"\nOverall Accuracy: {all_results['baseline']['accuracy']:.4f}")
    print(f"Overall F1: {all_results['baseline']['f1']:.4f}")
    print(f"AUC: {all_results['baseline']['auc']:.4f}")
    print(f"Demographic Parity Max Diff: {fairness_base.get('demographic_parity_diff_max', 'N/A'):.4f}" if isinstance(fairness_base.get('demographic_parity_diff_max'), float) else f"Demographic Parity Max Diff: N/A")
    print(f"Equalized Odds Max: {fairness_base.get('equalized_odds_max', 'N/A'):.4f}" if isinstance(fairness_base.get('equalized_odds_max'), float) else f"Equalized Odds Max: N/A")

    # =====================================================
    # 2) Independence (Demographic Parity)
    # =====================================================
    print("\n" + "="*60)
    print("2. INDEPENDENCE (Demographic Parity)")
    print("="*60)
    
    try:
        thresh_indep = ThresholdOptimizer(
            estimator=base_model,
            constraints="demographic_parity",
            predict_method="predict_proba",
            prefit=True,
            grid_size=1000,  # More granular grid search
        )
        thresh_indep.fit(X_test, y_test, sensitive_features=race_test)
        y_pred_indep = thresh_indep.predict(X_test, sensitive_features=race_test)

        df_indep = compute_group_table(y_test, y_pred_indep, race_test, "demographic_parity")
        df_indep.to_csv(OUT_DIR / "group_metrics_demographic_parity.csv", index=False)
        
        # Compute ROC for independence
        # Note: We still use original probs for ROC, as threshold optimization doesn't change probabilities
        roc_indep = compute_roc_points(y_test, probs_test, race_test, "independence")
        all_roc_points.append(roc_indep)
        
        fairness_indep = compute_fairness_metrics(df_indep)
        all_results["independence"] = {
            "accuracy": float(accuracy_score(y_test, y_pred_indep)),
            "f1": float(f1_score(y_test, y_pred_indep)),
            "fairness": fairness_indep,
        }
        
        print(df_indep.to_string(index=False))
        print(f"\nOverall Accuracy: {all_results['independence']['accuracy']:.4f}")
        print(f"Overall F1: {all_results['independence']['f1']:.4f}")
        print(f"Demographic Parity Max Diff: {fairness_indep.get('demographic_parity_diff_max', 'N/A'):.4f}" if isinstance(fairness_indep.get('demographic_parity_diff_max'), float) else f"Demographic Parity Max Diff: N/A")
        print(f"Equalized Odds Max: {fairness_indep.get('equalized_odds_max', 'N/A'):.4f}" if isinstance(fairness_indep.get('equalized_odds_max'), float) else f"Equalized Odds Max: N/A")
    except Exception as e:
        print(f"Error with demographic parity: {e}")
        all_results["independence"] = {"error": str(e)}

    # =====================================================
    # 3) Separation (Equalized Odds)
    # =====================================================
    print("\n" + "="*60)
    print("3. SEPARATION (Equalized Odds)")
    print("="*60)
    
    try:
        thresh_sep = ThresholdOptimizer(
            estimator=base_model,
            constraints="equalized_odds",
            predict_method="predict_proba",
            prefit=True,
            grid_size=1000,  # More granular grid search
        )
        thresh_sep.fit(X_test, y_test, sensitive_features=race_test)
        y_pred_sep = thresh_sep.predict(X_test, sensitive_features=race_test)

        df_sep = compute_group_table(y_test, y_pred_sep, race_test, "equalized_odds")
        df_sep.to_csv(OUT_DIR / "group_metrics_equalized_odds.csv", index=False)
        
        # Compute ROC for separation
        roc_sep = compute_roc_points(y_test, probs_test, race_test, "separation")
        all_roc_points.append(roc_sep)
        
        fairness_sep = compute_fairness_metrics(df_sep)
        all_results["separation"] = {
            "accuracy": float(accuracy_score(y_test, y_pred_sep)),
            "f1": float(f1_score(y_test, y_pred_sep)),
            "fairness": fairness_sep,
        }
        
        print(df_sep.to_string(index=False))
        print(f"\nOverall Accuracy: {all_results['separation']['accuracy']:.4f}")
        print(f"Overall F1: {all_results['separation']['f1']:.4f}")
        print(f"Demographic Parity Max Diff: {fairness_sep.get('demographic_parity_diff_max', 'N/A'):.4f}" if isinstance(fairness_sep.get('demographic_parity_diff_max'), float) else f"Demographic Parity Max Diff: N/A")
        print(f"Equalized Odds Max: {fairness_sep.get('equalized_odds_max', 'N/A'):.4f}" if isinstance(fairness_sep.get('equalized_odds_max'), float) else f"Equalized Odds Max: N/A")
    except Exception as e:
        print(f"Error with equalized odds: {e}")
        all_results["separation"] = {"error": str(e)}

    # =====================================================
    # Save ROC points
    # =====================================================
    df_roc_all = pd.concat(all_roc_points, ignore_index=True)
    df_roc_all.to_csv(OUT_DIR / "roc_points.csv", index=False)
    print(f"\nROC points saved to: {OUT_DIR / 'roc_points.csv'}")

    # =====================================================
    # Save comprehensive results
    # =====================================================
    with open(OUT_DIR / "all_fairness_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary comparison table
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    summary_rows = []
    for method in ["baseline", "independence", "separation"]:
        if method in all_results and "error" not in all_results[method]:
            res = all_results[method]
            fairness = res['fairness']
            summary_rows.append({
                "Method": method,
                "Accuracy": f"{res['accuracy']:.4f}",
                "F1": f"{res['f1']:.4f}",
                "DP_Max_Diff": f"{fairness.get('demographic_parity_diff_max', float('nan')):.4f}",
                "EqOdds_Max": f"{fairness.get('equalized_odds_max', float('nan')):.4f}",
            })
    
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(OUT_DIR / "fairness_comparison_summary.csv", index=False)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()