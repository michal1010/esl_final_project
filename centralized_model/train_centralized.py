from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainConfig:
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    lr: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 200
    patience: int = 15
    threshold: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def infer_continuous_columns(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    cont = []
    for c in feature_cols:
        # only numeric columns
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        nunique = df[c].nunique(dropna=True)
        if nunique > 5:
            cont.append(c)
    return cont


class LogisticModel(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def eval_loss(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: TrainConfig,
    device: str,
) -> Tuple[torch.nn.Module, dict]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        tr_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        va_loss = eval_loss(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_loss": best_val, "epochs_ran": len(history["train_loss"]), "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default="/mnt/data/compas/propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv",
        help="Path to propublica_data_for_fairml.csv",
    )
    ap.add_argument("--out", type=str, default="/mnt/data/centralized_out", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument(
        "--drop_race_from_features",
        action="store_true",
        help="Fairness-through-unawareness baseline: drop race columns from model inputs (still kept for evaluation).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, patience=args.patience, lr=args.lr, batch_size=args.batch_size)
    set_seed(cfg.seed)

    df = pd.read_csv(args.data)

    label_col = "Two_yr_Recidivism"
    if label_col not in df.columns:
        raise ValueError(f"Expected label column '{label_col}' in CSV. Found columns: {list(df.columns)}")
    race_cols = [
        "African_American",
        "Asian",
        "Hispanic",
        "Native_American",
        "Other",
    ]
    missing_race = [c for c in race_cols if c not in df.columns]
    if missing_race:
        raise ValueError(f"Missing expected race columns {missing_race} in CSV.")

    feature_cols = [c for c in df.columns if c != label_col]

    if args.drop_race_from_features:
        feature_cols = [c for c in feature_cols if c not in race_cols]

    X_df = df[feature_cols].copy()
    for c in X_df.columns:
        if not np.issubdtype(X_df[c].dtype, np.number):
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    if X_df.isna().any().any():
        X_df = X_df.fillna(0)

    y = df[label_col].astype(int).to_numpy()

    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X_df.to_numpy(),
        y,
        np.arange(len(df)),
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_trainval,
        y_trainval,
        idx_trainval,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=y_trainval,
    )

    cont_cols = infer_continuous_columns(df, feature_cols)
    cont_idx = [feature_cols.index(c) for c in cont_cols]

    scaler = StandardScaler()
    if cont_idx:
        scaler.fit(X_train[:, cont_idx])
        for split in (X_train, X_val, X_test):
            split[:, cont_idx] = scaler.transform(split[:, cont_idx])
    else:
        scaler.fit(np.zeros((1, 1)))

    train_loader = make_dataloader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, cfg.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LogisticModel(n_features=X_train.shape[1])

    model, train_info = train(model, train_loader, val_loader, cfg, device)

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    joblib.dump(scaler, os.path.join(args.out, "scaler.joblib"))

    meta = {
        "data": os.path.abspath(args.data),
        "out": os.path.abspath(args.out),
        "feature_cols": feature_cols,
        "continuous_cols_scaled": cont_cols,
        "label_col": label_col,
        "race_cols": race_cols,
        "drop_race_from_features": bool(args.drop_race_from_features),
        "splits": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
        },
        "config": asdict(cfg),
        "train_info": train_info,
    }

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    np.savez(
        os.path.join(args.out, "split_indices.npz"),
        train_idx=idx_train,
        val_idx=idx_val,
        test_idx=idx_test,
    )

    print("Saved model + scaler + config to:", args.out)


if __name__ == "__main__":
    main()
