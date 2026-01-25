import joblib
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from sklearnexample.task import (
    create_logreg_model,
    set_initial_params,
    get_model_params,
    set_model_params,
    NUMERIC_FEATURES, EXPECTED_DUMMY_COLS
)

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
 
    num_rounds: int = context.run_config["num-server-rounds"]

    # Initialize LogisticRegression model parameters
    model = create_logreg_model()
    n_features = len(NUMERIC_FEATURES + EXPECTED_DUMMY_COLS)
    set_initial_params(model, n_features=n_features)
    
    arrays = ArrayRecord(get_model_params(model))

    # FedAvg strategy
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # Start FedAvg
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    # Update server model with final aggregated parameters
    ndarrays = result.arrays.to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Save model
    joblib.dump(model, "logreg_model.pkl")
    print("Final model saved to logreg_model.pkl")
