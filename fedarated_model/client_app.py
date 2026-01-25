
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

    y_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_proba, labels=UNIQUE_LABELS)
    acc = model.score(X_test, y_test)

    metrics = {
        "num-examples": len(X_test),
        "accuracy": acc,
        "loss": loss,
    }

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )
