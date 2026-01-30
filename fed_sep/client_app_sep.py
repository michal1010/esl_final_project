import os
import csv
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import confusion_matrix, log_loss
from flwr.common import ArrayRecord, logger

from task import (
    create_logreg_model,
    get_model_params,
    set_model_params,
    set_initial_params,
    load_data_by_cid,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    UNIQUE_LABELS, calculate_roc_metrics, calculate_PPV_NPV
)
import numpy as np
flag=0
app = ClientApp()

def tpr_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr
#Training
@app.train()
def train(msg: Message, context: Context):
    cid = context.node_config["partition-id"]

    X_train, y_train, _, _,sample_weights = load_data_by_cid(cid)
    sample_weights = np.clip(sample_weights, 0.1, 10.0)
    model = create_logreg_model()
    set_initial_params(model, n_features=X_train.shape[1])

    # Receive global params
    params = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, params)
    if(flag==1):
        model.fit(X_train, y_train,sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_train)
    loss = log_loss(y_train, y_proba, labels=UNIQUE_LABELS)
    if(flag==1):
        acc = model.score(X_train, y_train,sample_weight=sample_weights)
    else:
       acc = model.score(X_train, y_train)
    
    arrays = get_model_params(model)
    array_record = ArrayRecord(arrays)
    metrics = {
        "num-examples": len(X_train),
        "train_loss": loss,
        "train_accuracy": acc,
    }

    return Message(
        content=RecordDict({
            "arrays": array_record,
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )
# Helper function to obtain the optimal threshold per race to satisfy separation
def find_threshold_for_equalized_odds(y_true, y_probs, target_tpr, target_fpr, steps=101, relaxed=True):

    best_t = 0.5
    best_dist = float("inf")
    best_tpr = None
    best_fpr = None
    for t in np.linspace(0, 1, steps):
        y_pred = (y_probs >= t).astype(int)
        tpr, fpr = tpr_fpr(y_true, y_pred)
        if relaxed:
            dist = (tpr - target_tpr) ** 2
        else:
            dist = (tpr - target_tpr) ** 2 + (fpr - target_fpr) ** 2
        if dist < best_dist:
            logger.log(level=20, msg=f"Found a better threshold: {dist}, with threshold: {t}, and tpr: {tpr} (target = {target_tpr}")
            best_dist = dist
            best_t = t
            best_tpr = tpr
            best_fpr = fpr
    return best_t, best_tpr, best_fpr

#Evaluation
@app.evaluate()
def evaluate(msg: Message, context: Context):
    save_thresholds = {}
    race_stat={}
    cid = context.node_config["partition-id"]
    _, _, X_test, y_test,sample_weights = load_data_by_cid(cid)

    model = create_logreg_model()
    set_initial_params(model, n_features=X_test.shape[1])

    # Set aggregated weights
    params = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, params)

    # Raw predicted probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Step 1: Identify indices for each race
    race_dict = {race: [] for race in CATEGORICAL_FEATURES['race']}
    race_start_idx = len(NUMERIC_FEATURES)
    for i, race in enumerate(CATEGORICAL_FEATURES['race']):
        race_col = race_start_idx + i
        race_dict[race] = np.where(X_test[:, race_col] == 1)[0]

    calculate_roc_metrics(y_test, race_dict, y_proba)

    # Step 2: Compute overall TPR and FPR
    overall_pred = (y_proba >= 0.5).astype(int)
    overall_tpr, overall_fpr = tpr_fpr(y_test, overall_pred)

    calculate_PPV_NPV(y_test, race_dict, overall_pred)

    # Step 3: Adjust thresholds per race for true separation (equal TPR & FPR)
    y_pred_adj = np.zeros_like(y_test)
    for race, indices in race_dict.items():
        if len(indices) == 0:
            continue
        threshold, best_tpr, best_fpr = find_threshold_for_equalized_odds(
            y_test[indices],
            y_proba[indices],
            target_tpr=overall_tpr,
            target_fpr=overall_fpr,
            relaxed=False
        )
        print(f"Race: {race}, Threshold: {threshold:.3f}, TPR: {best_tpr:.3f}, FPR: {best_fpr:.3f}")
        race_stat[race]={"TPR":best_tpr, "FPR":best_fpr, "Threshold":threshold}
        y_pred_adj[indices] = (y_proba[indices] >= threshold).astype(int)
        new_tpr, new_fpr = tpr_fpr(y_test[indices], y_pred_adj[indices])
        print(f"New TPR: {new_tpr:.3f}, New FPR: {new_fpr:.3f}")

    # Step 4: Metrics after adjustment
    loss = log_loss(y_test, y_proba, labels=UNIQUE_LABELS)
    acc = np.mean(y_pred_adj == y_test)
    # Per-race metrics
    race_metrics = {}
    for race, indices in race_dict.items():
        if len(indices) == 0:
            race_metrics[race] = {"TPR": None, "FPR": None}
            continue
        race_metrics[race] = {"Race":race, "TPR": race_stat[race]["TPR"], "FPR": race_stat[race]["FPR"], "no_of_samples":len(indices), "Threshold": race_stat[race]["Threshold"]}
        print(race_metrics)
    
    with open("metrics.csv", "w", newline="") as f:
        fieldnames = ["Race", "TPR", "FPR", "no_of_samples", "Threshold"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(race_metrics.values())
    
    metrics = {
        "num-examples": len(y_test),
        "accuracy": acc,
        "loss": loss,
    }

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )