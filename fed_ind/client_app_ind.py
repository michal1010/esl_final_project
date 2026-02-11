from flwr.app import Context, Message, MetricRecord, RecordDict
from fairlearn.postprocessing import ThresholdOptimizer
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss
from flwr.common import ArrayRecord
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task import (
    create_logreg_model,
    get_model_params,
    set_model_params,
    set_initial_params,
    load_data_by_cid,
    UNIQUE_LABELS,
    CATEGORICAL_FEATURES
)
flag=1
app = ClientApp()

# Train
@app.train()
def train(msg: Message, context: Context):
    cid = context.node_config["partition-id"]
    X_train, y_train, _, _, sample_weights = load_data_by_cid(cid)
    sample_weights = np.clip(sample_weights, 0.1, 10.0)

    model = create_logreg_model()
    set_initial_params(model, n_features=X_train.shape[1])

    # Receive global parameters
    params = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, params)

    # Fit with or without sample weights
    if flag == 1:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    # Metrics
    y_proba = model.predict_proba(X_train)
    loss = log_loss(y_train, y_proba, labels=UNIQUE_LABELS)
    if flag == 1:
        acc = model.score(X_train, y_train, sample_weight=sample_weights)
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

# Evaluate
@app.evaluate()
def evaluate(msg: Message, context: Context):
    cid = context.node_config["partition-id"]
    _, _, X_test, y_test, _ = load_data_by_cid(cid)

    model = create_logreg_model()
    set_initial_params(model, n_features=X_test.shape[1])

    # Load aggregated FL parameters
    params = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, params)

    # Extract sensitive attribute (race)
    race_start_idx = X_test.shape[1] - len(CATEGORICAL_FEATURES["race"])
    sensitive_features = np.argmax(X_test[:, race_start_idx:], axis=1)

    # Apply Fairlearn post-processing
    threshold_model = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity", 
        predict_method="predict_proba"
)
    threshold_model.fit(X_test, y_test, sensitive_features=sensitive_features)
    y_pred_fair = threshold_model.predict(X_test, sensitive_features=sensitive_features)

    # Metrics
    y_proba = model.predict_proba(X_test)[:, 1]
    loss = log_loss(y_test, y_proba, labels=UNIQUE_LABELS)
    acc = np.mean(y_pred_fair == y_test)

    metrics = {
        "num-examples": len(y_test),
        "accuracy": acc,
        "loss": loss,
    }

    # Plot boxplots for independence check
    race_dict = {race: [] for race in CATEGORICAL_FEATURES['race']}
    for idx, row in enumerate(X_test):
        for i, race in enumerate(CATEGORICAL_FEATURES['race']):
            if row[race_start_idx + i] == 1:
                race_dict[race].append(idx)
                break
    y_proba = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(8, 6))
    for race, indices in race_dict.items():
            if len(indices) == 0:
                continue
            # Raw predictions
            sns.kdeplot(
                y_proba[indices],
                label=f"{race} (raw)",
                linestyle='--'
            )
            # Postprocessed predictions
            sns.kdeplot(
                y_pred_fair[indices],
                label=f"{race} (postprocessed)"
            )

    plt.xlabel("Predicted probability P(ŷ = 1)")
    plt.ylabel("Density")
    plt.title("Raw vs Postprocessed probabilities per race (Demographic Parity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("postprocessing_kde.png")
    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )