# Federated Learning on COMPAS Dataset

This project evaluates the fairness impacts of federated learning versus centralized learning on the COMPAS recidivism dataset using the Flower federated learning framework.

## Project Structure

```
.
├── fed_ind/                    # Independent data partitioning experiment
│   ├── client_app_ind.py
│   ├── server_app_ind.py
│   ├── task.py
│   └── pyproject.toml
│
├── fed_sep/                    # Dirichlet distribution partitioning
│   ├── client_app_ind.py
│   ├── server_app_ind.py
│   ├── task.py
│   └── pyproject.toml Dirichlet
│   └── pyproject.toml
│
└── data_analytics/             # Data preparation and 
federated_data_random/  # Data partitioned randomly

analysis scripts
```

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Experiment Descriptions

### 1. **fed_ind** - Random Equal Splitting
Cross-silo federated learning with data randomly and equally distributed across 5 clients.

### 2. **fed_sep** - Dirichlet Distribution
Cross-silo federated learning with data distributed using Dirichlet distribution across 5 clients (simulates non-IID data).

Both experiments use:
- 5 federated clients (cross-silo architecture)
- 25% of original dataset reserved for testing
- Logistic regression with SAGA solver
- FedAvg aggregation strategy

---

## Installation & Setup

### Option 1: Random Equal Splitting (fed_ind)

```bash
# Navigate to the experiment directory
cd fed_ind

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\activate   # On Windows

# Install dependencies from pyproject.toml
pip install -e .

# Run the federated learning experiment
flwr run .
```

### Option 2: Dirichlet Distribution (fed_sep)

```bash
# Navigate to the experiment directory
cd fed_sep

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\activate   # On Windows

# Install dependencies from pyproject.toml
pip install -e .

# Run the federated learning experiment
flwr run .
```

---

## Configuration

Each experiment can be configured via `pyproject.toml`:

```toml
[tool.flwr.app.config]
penalty = "l2"                    # Regularization penalty
num-server-rounds = 5             # Number of federated rounds
min-available-clients = 2         # Minimum clients needed

[tool.flwr.federations.local-simulation]
options.num-supernodes = 5        # Number of federated clients
```

To modify:
1. Edit `pyproject.toml` in the respective experiment directory
2. Adjust parameters as needed
3. Re-run `flwr run .`

---

## Output

After successful execution, each experiment produces:
- `logreg_model.pkl` - Trained federated model
- `metrics.csv` - Training and evaluation metrics per round
- Console output with training progress and accuracy

---

## Dataset

**COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)**

**Features:**
- `sex`, `age`, `race` (categorical)
- `priors_count`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `decile_score` (numeric)

**Target:**
- `is_recid` (binary: 0 = no recidivism, 1 = recidivism)

**Data Splits:**
- **fed_ind**: Random equal distribution across 5 clients
- **fed_sep**: Dirichlet-based distribution (non-IID)
- **Test set**: 25% held out for evaluation (same for all clients)

---

## Deactivating Virtual Environment

When finished:
```bash
deactivate
```

---

## Troubleshooting

### "No module named 'sklearnexample'"
```bash
# Ensure you're in the correct directory and run:
pip install -e .
```

### "ModuleNotFoundError: No module named 'flwr'"
```bash
# Install dependencies:
pip install -e .
```

### Data file not found
Ensure `federated_data_random/` directory exists with:
- `client_random_1.csv` through `client_random_5.csv`
- `client_random_test.csv`

### Wrong Python version
```bash
# Check Python version (should be 3.8+)
python3 --version
```

---

## Research Context

This project compares:
1. **Federated Learning** (distributed training across clients)
2. **Centralized Learning** (baseline for comparison)

**Evaluation Focus:**
- Model accuracy
- Fairness metrics across demographic groups (race, sex)
- Impact of data distribution strategies on bias

---

## License

Apache-2.0

---

## Dependencies

Core dependencies (automatically installed via `pip install -e .`):
- `flwr[simulation]>=1.24.0` - Flower federated learning framework
- `scikit-learn>=1.6.1` - Machine learning models
- `flwr-datasets[vision]>=0.5.0` - Dataset utilities
- `pandas` - Data manipulation
- `numpy` - Numerical computing

See `pyproject.toml` for complete dependency list.

---

## Quick Start Summary

```bash
# For Random Equal Split experiment:
cd fed_ind && python3 -m venv venv && source venv/bin/activate && pip install -e . && flwr run .

# For Dirichlet Distribution experiment:
cd fed_sep && python3 -m venv venv && source venv/bin/activate && pip install -e . && flwr run .
```