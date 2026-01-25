import json
import warnings
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss
from flwr.common import ArrayRecord
from task import (
    create_logreg_model,
    get_model_params,
    set_model_params,
    set_initial_params,
    load_data_by_cid,
    UNIQUE_LABELS,
    CATEGORICAL_FEATURES
)

app = ClientApp()

# Train
@app.train()
def train(msg: Message, context: Context):
    cid = context.node_config["partition-id"]

    X_train, y_train, _, _ = load_data_by_cid(cid)
    model = create_logreg_model()
    set_initial_params(model, n_features=X_train.shape[1])

    # Receive global params
    params = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, params)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_train)
    loss = log_loss(y_train, y_proba, labels=UNIQUE_LABELS)
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

    _, _, X_test, y_test = load_data_by_cid(cid)
    model = create_logreg_model()
    set_initial_params(model, n_features=X_test.shape[1])

    params = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, params)

    race_dict = {race : [] for race in CATEGORICAL_FEATURES['race']}
    for index, row in enumerate(X_test):

        for i, race in enumerate(CATEGORICAL_FEATURES['race']):
            if row[i - 6]: # the index of the chosen race
                race_dict[race].append(index)
                break


    y_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_proba, labels=UNIQUE_LABELS)
    acc = model.score(X_test, y_test)

    y_pred = (y_proba[:, 1] >= 0.5).astype(int)
    race_metrics = {}

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

        race_metrics[race] = {
            "num_examples": len(indices),
            "TPR": tpr,
            "FPR": fpr
        }

    with open(f'race_metrics{cid}.json', 'w') as f:
        json.dump(race_metrics, f)


    metrics = {
        "num-examples": len(X_test),
        "accuracy": acc,
        "loss": loss,
        "Other_TPR": race_metrics['Other']['TPR'],
        "Caucasian_TPR": race_metrics['Caucasian']['TPR'],
        "African-American_TPR": race_metrics['African-American']['TPR'],
        "Hispanic_TPR": race_metrics['Hispanic']['TPR'],
        "Asian_TPR": race_metrics['Asian']['TPR'],
        "Native American_TPR": race_metrics['Native American']['TPR'],
        "Other_FPR": race_metrics['Other']['FPR'],
        "Caucasian_FPR": race_metrics['Caucasian']['FPR'],
        "African-American_FPR": race_metrics['African-American']['FPR'],
        "Hispanic_FPR": race_metrics['Hispanic']['FPR'],
        "Asian_FPR": race_metrics['Asian']['FPR'],
        "Native American_FPR": race_metrics['Native American']['FPR'],
   }

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )
