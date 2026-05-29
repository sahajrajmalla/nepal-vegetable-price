"""
dl_models.py — Deep Learning Forecasting Models (LSTM & GRU) — PyTorch
========================================================================

Implements sequence-to-one recurrent neural networks:

    1. **LSTM** — Long Short-Term Memory network
    2. **GRU**  — Gated Recurrent Unit network

Both architectures:
    • Accept multi-feature sequences (shape: [samples, timesteps, features])
    • Support stacked layers with dropout regularisation
    • Use early stopping with patience to prevent overfitting
    • Apply MinMax or Standard scaling (inverted after prediction)
    • Are fully deterministic (seed propagation)
    • Save training history (loss, val_loss) as CSV for research reporting

References
----------
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
  Neural Computation, 9(8), 1735–1780.
- Cho, K., et al. (2014). Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation. EMNLP.

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import copy
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger("kalimati.models.dl")


# ═════════════════════════════════════════════════════════════════════════════
# 1. SEQUENCE PREPARATION
# ═════════════════════════════════════════════════════════════════════════════


def create_sequences(
    data: np.ndarray,
    seq_length: int,
    target_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D array into overlapping sequences for RNN input.

    Parameters
    ----------
    data : np.ndarray
        Scaled data of shape (n_samples, n_features).
    seq_length : int
        Number of time steps per input sequence.
    target_idx : int
        Column index of the target variable in ``data``.

    Returns
    -------
    tuple
        X: shape (n_sequences, seq_length, n_features)
        y: shape (n_sequences,)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)


def prepare_dl_data(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    target: str = "Average",
    feature_cols: Optional[List[str]] = None,
    scaler_type: str = "minmax",
) -> Dict[str, Any]:
    """
    Prepare train/test sequences for DL models.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame for one commodity (sorted by date).
    cfg : dict
        Pipeline configuration.
    target : str
        Target column name.
    feature_cols : list of str, optional
        Feature columns to include. If None, uses target only (univariate).
    scaler_type : str
        'minmax' or 'standard'.

    Returns
    -------
    dict
        'X_train', 'y_train', 'X_test', 'y_test',
        'scaler', 'target_idx', 'seq_length',
        'train_dates', 'test_dates'
    """
    seq_length = cfg["models"]["dl"]["lstm"]["sequence_length"]

    df = df.sort_values("Date").reset_index(drop=True)

    # Select columns for the model
    if feature_cols is None:
        cols = [target]
    else:
        cols = [target] + [c for c in feature_cols if c != target]

    # Target index in the feature array
    target_idx = 0  # target is always the first column

    # Split FIRST by date (same cutoff as evaluation.fixed_split), THEN
    # drop NaN within each partition.  This ensures the DL test set matches
    # the test set used by ML/statistical models — critical for fair
    # model comparison in the research paper.
    df_selected = df[["Date"] + cols].copy()
    train_end = pd.Timestamp(cfg["evaluation"]["train_end_date"])
    data_max = df_selected["Date"].max()

    if train_end >= data_max:
        split_idx = int(len(df_selected) * 0.80)
        train_end = df_selected.iloc[split_idx]["Date"]
        logger.warning(
            f"DL split: configured date exceeds data range. "
            f"Auto-adapting to 80/20 at {train_end.date()}."
        )

    train_part = df_selected[df_selected["Date"] <= train_end].dropna().reset_index(drop=True)
    test_part = df_selected[df_selected["Date"] > train_end].dropna().reset_index(drop=True)

    train_data = train_part[cols].values
    test_data = test_part[cols].values
    train_dates = train_part["Date"].values
    test_dates = test_part["Date"].values

    if len(train_data) < seq_length + 10:
        raise ValueError(
            f"Insufficient training data: {len(train_data)} rows "
            f"(need ≥ {seq_length + 10})"
        )

    # Scale
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_data)

    # For test data, combine last `seq_length` rows of train + test
    # to create continuous sequences
    combined = np.vstack([train_data[-seq_length:], test_data])
    combined_scaled = scaler.transform(combined)

    # Create sequences
    X_train, y_train = create_sequences(train_scaled, seq_length, target_idx)
    X_test, y_test = create_sequences(combined_scaled, seq_length, target_idx)

    logger.info(
        f"DL data prepared: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"seq_length={seq_length}, features={len(cols)}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "target_idx": target_idx,
        "seq_length": seq_length,
        "feature_cols": cols,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════


def _set_dl_seeds(seed: int = 42) -> None:
    """Set all DL-related random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMForecaster(nn.Module):
    """Stacked LSTM for time-series forecasting."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list = None,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 128]

        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes

        # Build stacked LSTM layers
        self.lstms = nn.ModuleList()
        input_size = n_features
        for i in range(n_layers):
            h = hidden_sizes[i] if i < len(hidden_sizes) else hidden_sizes[-1]
            self.lstms.append(nn.LSTM(
                input_size=input_size, hidden_size=h,
                num_layers=1, batch_first=True, dropout=0,
            ))
            input_size = h

        self.dropout = nn.Dropout(dropout)
        last_h = hidden_sizes[-1] if n_layers <= len(hidden_sizes) else hidden_sizes[-1]
        self.fc1 = nn.Linear(last_h, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"LSTM model built: {total:,} parameters")

    def forward(self, x):
        out = x
        for lstm in self.lstms:
            out, _ = lstm(out)
            out = self.dropout(out)
        out = out[:, -1, :]  # Take last time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


class GRUForecaster(nn.Module):
    """Stacked GRU for time-series forecasting."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list = None,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 128]

        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes

        self.grus = nn.ModuleList()
        input_size = n_features
        for i in range(n_layers):
            h = hidden_sizes[i] if i < len(hidden_sizes) else hidden_sizes[-1]
            self.grus.append(nn.GRU(
                input_size=input_size, hidden_size=h,
                num_layers=1, batch_first=True, dropout=0,
            ))
            input_size = h

        self.dropout = nn.Dropout(dropout)
        last_h = hidden_sizes[-1] if n_layers <= len(hidden_sizes) else hidden_sizes[-1]
        self.fc1 = nn.Linear(last_h, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"GRU model built: {total:,} parameters")

    def forward(self, x):
        out = x
        for gru in self.grus:
            out, _ = gru(out)
            out = self.dropout(out)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ═════════════════════════════════════════════════════════════════════════════


def train_dl_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Dict[str, Any],
    model_type: str = "lstm",
    validation_split: float = 0.15,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train a PyTorch DL model with early stopping, LR scheduler, and tqdm.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    X_train : np.ndarray
        Training sequences of shape (n, seq_len, n_features).
    y_train : np.ndarray
        Training targets of shape (n,).
    cfg : dict
        Pipeline configuration.
    model_type : str
        'lstm' or 'gru' (for reading config).
    validation_split : float
        Fraction of training data for validation.

    Returns
    -------
    tuple
        (trained_model, history_dict) where history_dict has keys
        'train_loss', 'val_loss', 'lr'.
    """
    dl_cfg = cfg["models"]["dl"][model_type]
    epochs = dl_cfg.get("epochs", 150)
    batch_size = dl_cfg.get("batch_size", 32)
    patience = dl_cfg.get("patience", 15)
    lr = dl_cfg.get("learning_rate", 0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Split into train/val
    n_val = int(len(X_train) * validation_split)
    n_train = len(X_train) - n_val

    X_tr = torch.FloatTensor(X_train[:n_train]).to(device)
    y_tr = torch.FloatTensor(y_train[:n_train]).to(device)
    X_val = torch.FloatTensor(X_train[n_train:]).to(device)
    y_val = torch.FloatTensor(y_train[n_train:]).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    logger.info(
        f"Training {model_type.upper()}: epochs={epochs}, "
        f"batch_size={batch_size}, patience={patience}, device={device}"
    )

    # Training loop with early stopping
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    pbar = tqdm(range(epochs), desc=f"  {model_type.upper()} training", unit="epoch", leave=True)

    for epoch in pbar:
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Progress bar
        pbar.set_postfix({
            "loss": f"{avg_train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "lr": f"{current_lr:.1e}",
        })

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    pbar.close()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    actual_epochs = len(history["train_loss"])
    logger.info(
        f"{model_type.upper()} training complete: "
        f"{actual_epochs} epochs, best val_loss={best_val_loss:.6f}"
    )

    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION & INVERSE SCALING
# ═════════════════════════════════════════════════════════════════════════════


def predict_dl_model(
    model: nn.Module,
    X_test: np.ndarray,
    scaler: Any,
    target_idx: int = 0,
    n_features: int = 1,
) -> np.ndarray:
    """
    Generate predictions and inverse-transform to original scale.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    X_test : np.ndarray
        Test sequences.
    scaler : sklearn scaler
        Fitted scaler used during training.
    target_idx : int
        Target column index in the feature array.
    n_features : int
        Total number of features.

    Returns
    -------
    np.ndarray
        Predictions in original price scale.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        predictions_scaled = model(X_tensor).cpu().numpy()

    # Create a dummy array to inverse-transform just the target column
    dummy = np.zeros((len(predictions_scaled), n_features))
    dummy[:, target_idx] = predictions_scaled
    predictions_original = scaler.inverse_transform(dummy)[:, target_idx]

    return predictions_original


def inverse_transform_target(
    values: np.ndarray,
    scaler: Any,
    target_idx: int = 0,
    n_features: int = 1,
) -> np.ndarray:
    """Inverse-transform target values from scaled to original space."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, target_idx] = values
    return scaler.inverse_transform(dummy)[:, target_idx]


# ═════════════════════════════════════════════════════════════════════════════
# 5. UNIFIED DL RUNNER
# ═════════════════════════════════════════════════════════════════════════════


def predict_recursive_dl(
    model: nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Any,
    target_idx: int,
    seq_length: int,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Generate recursive multi-step forecasts for PyTorch LSTM or GRU."""
    from src.feature_engineering import engineer_features

    # Work on copies to prevent modifying caller's DataFrames
    train_df = train_df.copy()
    test_df = test_df.copy()

    target = cfg["preprocessing"]["target_column"]
    base_cols = ["Date", "Commodity", "Unit", "Minimum", "Maximum", target]

    for col in base_cols:
        if col not in train_df.columns:
            train_df[col] = "KVPI" if col == "Commodity" else ("Index" if col == "Unit" else train_df[target] if col in ["Minimum", "Maximum"] else None)
        if col not in test_df.columns:
            test_df[col] = "KVPI" if col == "Commodity" else ("Index" if col == "Unit" else test_df[target] if col in ["Minimum", "Maximum"] else None)

    train_base = train_df[base_cols].copy()
    test_base = test_df[base_cols].copy()

    combined = pd.concat([train_base, test_base], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    n_train = len(train_df)
    n_test = len(test_df)

    combined.loc[n_train:, target] = np.nan
    if "Minimum" in combined.columns:
        combined.loc[n_train:, "Minimum"] = np.nan
    if "Maximum" in combined.columns:
        combined.loc[n_train:, "Maximum"] = np.nan
        
    predictions = []

    device = next(model.parameters()).device
    model.eval()

    for i in range(n_test):
        idx = n_train + i
        
        # DL predicts the NEXT step (idx) based on the past seq_length steps.
        # So we only need the history up to idx - 1.
        sub_df = combined.iloc[:idx].copy()
        sub_df_feat = engineer_features(sub_df, cfg, commodity="KVPI")

        # Pick the columns used by DL features
        sub_df_model = sub_df_feat.reindex(columns=feature_cols, fill_value=0)

        # Scale last seq_length rows
        scaled_data = scaler.transform(sub_df_model.iloc[-seq_length:].values)

        # Reshape to (1, seq_length, n_features)
        X_seq = scaled_data.reshape(1, seq_length, -1)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(device)
            pred_scaled = model(X_tensor).cpu().numpy().item()

        # Inverse transform
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, target_idx] = pred_scaled
        pred_val = scaler.inverse_transform(dummy)[0, target_idx]

        predictions.append(pred_val)
        combined.loc[idx, target] = pred_val
        if "Minimum" in combined.columns:
            combined.loc[idx, "Minimum"] = pred_val
        if "Maximum" in combined.columns:
            combined.loc[idx, "Maximum"] = pred_val

    return np.array(predictions)


def _optuna_dl_objective(
    trial: "optuna.Trial",
    model_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    cfg: Dict[str, Any],
    model_type: str,
    seed: int,
) -> float:
    """
    Optuna objective function for DL hyperparameter optimisation.

    Searches over architecture (hidden sizes, layers, dropout) and
    training parameters (learning rate, batch size) using the same
    Bayesian TPE sampler used for tree-based models, ensuring a fair
    and scientifically rigorous comparison.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    model_class : type
        LSTMForecaster or GRUForecaster.
    X_train : np.ndarray
        Training sequences (n, seq_len, n_features).
    y_train : np.ndarray
        Training targets.
    n_features : int
        Number of input features per time step.
    cfg : dict
        Pipeline configuration.
    model_type : str
        'lstm' or 'gru'.
    seed : int
        Random seed.

    Returns
    -------
    float
        Best validation loss achieved during training.
    """
    _set_dl_seeds(seed)

    # ── Search space (mirrors the breadth of ML model Optuna search) ──
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = []
    for i in range(n_layers):
        h = trial.suggest_categorical(f"hidden_size_{i}", [32, 64, 128, 256])
        hidden_sizes.append(h)
    dropout = trial.suggest_float("dropout", 0.05, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # ── Build model ──
    model = model_class(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        n_layers=n_layers,
        dropout=dropout,
    )

    # ── Train with the trial's hyperparameters ──
    # Override config temporarily for this trial
    trial_cfg = copy.deepcopy(cfg)
    trial_cfg["models"]["dl"][model_type]["learning_rate"] = lr
    trial_cfg["models"]["dl"][model_type]["batch_size"] = batch_size
    # Use reduced patience for faster Optuna trials
    trial_cfg["models"]["dl"][model_type]["patience"] = min(
        10, trial_cfg["models"]["dl"][model_type].get("patience", 20)
    )

    trained_model, history = train_dl_model(
        model, X_train, y_train, trial_cfg, model_type=model_type,
    )

    # Return best validation loss
    best_val_loss = min(history["val_loss"])
    return best_val_loss


def run_dl_models(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all DL models (LSTM, GRU) with Optuna Bayesian
    hyperparameter optimisation.

    Parity with ML models
    ---------------------
    * **All features**: DL models now receive the same full feature set
      as tree-based ML models (no manual filtering).
    * **Optuna HPO**: Architecture and training hyperparameters are
      optimised using the same TPE sampler and trial budget as
      XGBoost/RandomForest, ensuring a scientifically fair comparison.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from src.evaluation import compute_all_metrics

    _set_dl_seeds(seed)

    target = cfg["preprocessing"]["target_column"]
    metrics_list = cfg.get("evaluation", {}).get("metrics", ["RMSE", "MAE", "MAPE", "sMAPE"])
    dl_cfg = cfg.get("models", {}).get("dl", {})
    results = {}

    # ── Use ALL features (same as ML models) for fair comparison ──
    # Previous versions restricted DL to ~36 manually selected features
    # while ML models used all ~67.  This created an unfair advantage
    # for tree-based models and would cause peer-review rejection.
    dl_features = feature_cols  # Pass through ALL features
    logger.info(
        f"DL models using ALL {len(dl_features) if dl_features else 0} features "
        f"(same as ML models for fair comparison)"
    )

    scaler_type = dl_cfg.get("lstm", {}).get("scaler", "minmax")

    try:
        data = prepare_dl_data(
            df, cfg, target=target,
            feature_cols=dl_features,
            scaler_type=scaler_type,
        )
    except ValueError as e:
        logger.error(f"DL data preparation failed: {e}")
        return results

    n_features = data["X_train"].shape[2]
    seq_length = data["seq_length"]

    # Reconstruct train_df and test_df for recursive forecasting
    train_dates = data["train_dates"]
    test_dates = data["test_dates"]
    train_df = df[df["Date"].isin(train_dates)].copy()
    test_df = df[df["Date"].isin(test_dates)].copy()

    strategy = cfg["evaluation"].get("strategy", "recursive")
    logger.info(f"Using forecasting strategy for DL: {strategy.upper()}")

    # ── Helper: run Optuna HPO + final training for a DL model ──
    def _train_with_optuna(model_class, model_type, model_cfg):
        """Run Optuna search, then retrain with best params."""
        n_trials = model_cfg.get("n_optuna_trials", 30)

        if n_trials > 0:
            logger.info(
                f"{model_type.upper()} Optuna search: {n_trials} trials "
                f"(hidden_sizes, n_layers, dropout, lr, batch_size)"
            )
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                lambda trial: _optuna_dl_objective(
                    trial, model_class,
                    data["X_train"], data["y_train"],
                    n_features, cfg, model_type, seed,
                ),
                n_trials=n_trials,
                show_progress_bar=True,
            )

            best = study.best_params
            best_val = study.best_value
            logger.info(
                f"{model_type.upper()} best Optuna params: {best}\n"
                f"  Best val loss: {best_val:.6f}"
            )

            # Extract best architecture
            best_n_layers = best["n_layers"]
            best_hidden = [best[f"hidden_size_{i}"] for i in range(best_n_layers)]
            best_dropout = best["dropout"]
            best_lr = best["learning_rate"]
            best_batch = best["batch_size"]

            del study
            import gc; gc.collect()
        else:
            # No Optuna — use config defaults (e.g., DEV mode)
            best_hidden = model_cfg.get("units", [64, 128])
            best_n_layers = model_cfg.get("n_layers", 2)
            best_dropout = model_cfg.get("dropout", 0.2)
            best_lr = model_cfg.get("learning_rate", 0.001)
            best_batch = model_cfg.get("batch_size", 32)

        # ── Retrain with best hyperparameters on full training data ──
        _set_dl_seeds(seed)
        final_model = model_class(
            n_features=n_features,
            hidden_sizes=best_hidden,
            n_layers=best_n_layers,
            dropout=best_dropout,
        )

        final_cfg = copy.deepcopy(cfg)
        final_cfg["models"]["dl"][model_type]["learning_rate"] = best_lr
        final_cfg["models"]["dl"][model_type]["batch_size"] = best_batch

        trained_model, history = train_dl_model(
            final_model, data["X_train"], data["y_train"],
            final_cfg, model_type=model_type,
        )
        return trained_model, history

    # ── LSTM ──
    if dl_cfg.get("lstm", {}).get("enabled", True):
        try:
            logger.info("── Training LSTM (with Optuna HPO) ──")

            lstm_cfg = dl_cfg.get("lstm", {})
            lstm_model, lstm_history = _train_with_optuna(
                LSTMForecaster, "lstm", lstm_cfg
            )

            if strategy == "recursive":
                logger.info("Generating recursive out-of-sample forecast for LSTM…")
                lstm_pred = predict_recursive_dl(
                    lstm_model, train_df, test_df, data["feature_cols"],
                    data["scaler"], data["target_idx"], seq_length, cfg
                )
                # Since create_sequences trims seq_length, we match dates
                y_test_original = inverse_transform_target(
                    data["y_test"], data["scaler"],
                    data["target_idx"], n_features
                )
                lstm_pred = lstm_pred[-len(y_test_original):]
            else:
                logger.info("Generating one-step-ahead rolling forecast for LSTM…")
                lstm_pred = predict_dl_model(
                    lstm_model, data["X_test"],
                    data["scaler"], data["target_idx"], n_features
                )
                y_test_original = inverse_transform_target(
                    data["y_test"], data["scaler"],
                    data["target_idx"], n_features
                )

            lstm_metrics = compute_all_metrics(
                y_test_original, lstm_pred, metrics_list
            )

            results["LSTM"] = {
                "predictions": lstm_pred,
                "y_test": y_test_original,
                "metrics": lstm_metrics,
                "model": lstm_model,
                "history": lstm_history,
                "scaler": data["scaler"],
            }
            rmse_val = lstm_metrics.get('RMSE')
            if rmse_val is not None:
                logger.info(f"LSTM — RMSE: {rmse_val:.4f}")
            else:
                logger.info(f"LSTM — RMSE: N/A")
        except Exception as e:
            logger.error(f"LSTM failed: {e}", exc_info=True)

    # ── GRU ──
    if dl_cfg.get("gru", {}).get("enabled", True):
        try:
            logger.info("── Training GRU (with Optuna HPO) ──")

            gru_cfg = dl_cfg.get("gru", {})
            gru_model, gru_history = _train_with_optuna(
                GRUForecaster, "gru", gru_cfg
            )

            if strategy == "recursive":
                logger.info("Generating recursive out-of-sample forecast for GRU…")
                gru_pred = predict_recursive_dl(
                    gru_model, train_df, test_df, data["feature_cols"],
                    data["scaler"], data["target_idx"], seq_length, cfg
                )
                y_test_original = inverse_transform_target(
                    data["y_test"], data["scaler"],
                    data["target_idx"], n_features
                )
                gru_pred = gru_pred[-len(y_test_original):]
            else:
                logger.info("Generating one-step-ahead rolling forecast for GRU…")
                gru_pred = predict_dl_model(
                    gru_model, data["X_test"],
                    data["scaler"], data["target_idx"], n_features
                )
                y_test_original = inverse_transform_target(
                    data["y_test"], data["scaler"],
                    data["target_idx"], n_features
                )

            gru_metrics = compute_all_metrics(
                y_test_original, gru_pred, metrics_list
            )

            results["GRU"] = {
                "predictions": gru_pred,
                "y_test": y_test_original,
                "metrics": gru_metrics,
                "model": gru_model,
                "history": gru_history,
                "scaler": data["scaler"],
            }
            rmse_val = gru_metrics.get('RMSE')
            if rmse_val is not None:
                logger.info(f"GRU — RMSE: {rmse_val:.4f}")
            else:
                logger.info(f"GRU — RMSE: N/A")
        except Exception as e:
            logger.error(f"GRU failed: {e}", exc_info=True)

    return results


def save_dl_model(
    model: nn.Module,
    cfg: Dict[str, Any],
    commodity: str,
    model_name: str,
) -> Path:
    """
    Save a PyTorch model to disk.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    cfg : dict
        Pipeline configuration.
    commodity : str
        Commodity name.
    model_name : str
        Model identifier.

    Returns
    -------
    Path
        Path to saved model file.
    """
    from src.utils import sanitize_commodity_name

    out_dir = Path(cfg["output"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_commodity_name(commodity)
    path = out_dir / f"{slug}_{model_name}.pt"
    torch.save(model.state_dict(), str(path))
    logger.info(f"Saved {model_name} for {commodity}: {path}")
    return path


def save_training_history(
    history: Dict[str, list],
    cfg: Dict[str, Any],
    commodity: str,
    model_name: str,
) -> Path:
    """
    Save training history (loss, val_loss, lr) as CSV for research reporting.

    Parameters
    ----------
    history : dict
        Training history with keys 'train_loss', 'val_loss', 'lr'.
    cfg : dict
        Pipeline configuration.
    commodity : str
        Commodity name.
    model_name : str
        Model identifier.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    from src.utils import sanitize_commodity_name

    out_dir = Path(cfg["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_commodity_name(commodity)

    df = pd.DataFrame({
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "learning_rate": history["lr"],
    })

    path = out_dir / f"{slug}_{model_name}_training_history.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved training history: {path}")
    return path
