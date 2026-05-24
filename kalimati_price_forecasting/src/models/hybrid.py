"""
hybrid.py — Hybrid Forecasting Models
=======================================

Implements two hybrid architectures that combine linear (ARIMA) and
nonlinear (LSTM/XGBoost) components:

    1. **ARIMA–LSTM Hybrid**: Fit ARIMA on the series, extract residuals,
       then train an LSTM on the residual sequence. Final forecast =
       ARIMA forecast + LSTM residual forecast.

    2. **ARIMA–XGBoost Hybrid**: Use the ARIMA in-sample fitted values
       as an additional feature alongside lag/calendar features, then
       train XGBoost on the augmented feature set.

Motivation
----------
Zhang (2003) demonstrated that hybrid models combining linear and
nonlinear components can outperform either component alone. The ARIMA
captures linear autocorrelation while the ML/DL component models
the nonlinear residual structure.

References
----------
- Zhang, G.P. (2003). Time series forecasting using a hybrid ARIMA
  and neural network model. Neurocomputing, 50, 159–175.
- Bousqaoui, H., Achchab, S., & Tikito, K. (2021). Machine learning
  applications in supply chains: Long short-term memory for demand
  forecasting. Applied Computational Intelligence and Soft Computing.

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import copy
import gc
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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger("kalimati.models.hybrid")


# ═════════════════════════════════════════════════════════════════════════════
# 1. ARIMA–LSTM HYBRID
# ═════════════════════════════════════════════════════════════════════════════


class ResidualLSTM(nn.Module):
    """Small LSTM for modelling ARIMA residuals."""

    def __init__(self, seq_length: int, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 64, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


def fit_arima_lstm_hybrid(
    train_y: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Fit a hybrid ARIMA + LSTM model.

    Pipeline
    --------
    1. Fit Auto-ARIMA on the training series.
    2. Extract in-sample residuals.
    3. Prepare residual sequences for LSTM.
    4. Train LSTM on residual sequences.
    5. Final forecast = ARIMA forecast + LSTM residual forecast.

    Parameters
    ----------
    train_y : array-like
        Training target values.
    test_y : array-like
        Test target values (for evaluation only).
    cfg : dict
        Pipeline configuration.
    seed : int
        Random seed.

    Returns
    -------
    dict
        'predictions': hybrid forecasts
        'arima_predictions': ARIMA component
        'lstm_predictions': LSTM residual component
        'metrics': evaluation metrics
        'arima_model': fitted ARIMA
        'lstm_model': fitted LSTM
    """
    from src.models.statistical import fit_auto_arima, predict_arima, get_arima_residuals
    from src.evaluation import compute_all_metrics
    from src.models.dl_models import create_sequences, _set_dl_seeds

    _set_dl_seeds(seed)

    train_y = np.asarray(train_y, dtype=float)
    test_y = np.asarray(test_y, dtype=float)
    horizon = len(test_y)
    metrics_list = cfg.get("evaluation", {}).get("metrics", ["RMSE", "MAE", "MAPE", "sMAPE"])

    logger.info("═" * 50)
    logger.info("  ARIMA–LSTM HYBRID MODEL")
    logger.info("═" * 50)

    # Step 1: Fit ARIMA
    logger.info("Step 1/4: Fitting ARIMA component …")
    arima_model = fit_auto_arima(train_y, cfg)

    # Step 2: Get ARIMA forecasts and residuals
    logger.info("Step 2/4: Extracting ARIMA residuals …")
    arima_output = predict_arima(arima_model, horizon, return_ci=False)
    arima_forecast = arima_output["predictions"]
    arima_residuals = get_arima_residuals(arima_model)

    # Step 3: Prepare residuals for LSTM
    logger.info("Step 3/4: Training LSTM on residuals …")
    seq_length = cfg["models"]["dl"]["lstm"]["sequence_length"]

    if len(arima_residuals) < seq_length + 10:
        logger.warning(
            f"Insufficient residuals ({len(arima_residuals)}) for LSTM. "
            f"Returning ARIMA-only forecast."
        )
        metrics = compute_all_metrics(test_y, arima_forecast, metrics_list)
        return {
            "predictions": arima_forecast,
            "arima_predictions": arima_forecast,
            "lstm_predictions": np.zeros(horizon),
            "metrics": metrics,
            "arima_model": arima_model,
            "lstm_model": None,
        }

    # Scale residuals
    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(arima_residuals.reshape(-1, 1))

    # Create sequences
    X_resid, y_resid = create_sequences(resid_scaled, seq_length, target_idx=0)

    # Build & train LSTM for residuals using PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = ResidualLSTM(
        seq_length=seq_length,
        dropout=cfg["models"]["dl"]["lstm"].get("dropout", 0.2),
    ).to(device)

    lstm_cfg = cfg["models"]["dl"]["lstm"]
    epochs = lstm_cfg.get("epochs", 150)
    batch_size = lstm_cfg.get("batch_size", 32)
    patience = lstm_cfg.get("patience", 15)
    lr = lstm_cfg.get("learning_rate", 0.001)

    # Split train/val
    n_val = int(len(X_resid) * 0.15)
    X_tr = torch.FloatTensor(X_resid[:-n_val]).to(device)
    y_tr = torch.FloatTensor(y_resid[:-n_val]).to(device)
    X_vl = torch.FloatTensor(X_resid[-n_val:]).to(device)
    y_vl = torch.FloatTensor(y_resid[-n_val:]).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    pbar = tqdm(range(epochs), desc="  Hybrid LSTM", unit="epoch", leave=True)
    for epoch in pbar:
        lstm_model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = lstm_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        lstm_model.eval()
        with torch.no_grad():
            val_loss = criterion(lstm_model(X_vl), y_vl).item()

        pbar.set_postfix({"loss": f"{np.mean(losses):.4f}", "val": f"{val_loss:.4f}"})

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(lstm_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    pbar.close()

    if best_state:
        lstm_model.load_state_dict(best_state)

    # Step 4: Forecast residuals iteratively
    logger.info("Step 4/4: Generating hybrid forecast …")

    lstm_model.eval()
    last_resid_seq = resid_scaled[-seq_length:].copy()
    lstm_resid_forecast = []

    with torch.no_grad():
        for _ in range(horizon):
            inp = torch.FloatTensor(last_resid_seq.reshape(1, seq_length, 1)).to(device)
            pred_scaled = lstm_model(inp).cpu().item()
            lstm_resid_forecast.append(pred_scaled)
            last_resid_seq = np.roll(last_resid_seq, -1, axis=0)
            last_resid_seq[-1, 0] = pred_scaled

    lstm_resid_forecast = np.array(lstm_resid_forecast).reshape(-1, 1)
    lstm_resid_original = scaler.inverse_transform(lstm_resid_forecast).flatten()

    # Hybrid forecast = ARIMA + LSTM residuals
    hybrid_forecast = arima_forecast + lstm_resid_original

    metrics = compute_all_metrics(test_y, hybrid_forecast, metrics_list)

    logger.info(f"ARIMA–LSTM Hybrid — RMSE: {metrics.get('RMSE', 'N/A'):.4f}")

    return {
        "predictions": hybrid_forecast,
        "arima_predictions": arima_forecast,
        "lstm_predictions": lstm_resid_original,
        "metrics": metrics,
        "arima_model": arima_model,
        "lstm_model": lstm_model,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. ARIMA–XGBOOST HYBRID
# ═════════════════════════════════════════════════════════════════════════════


def predict_recursive_arima_xgb(
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    arima_forecast: np.ndarray,
    arima_fitted: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Generate recursive multi-step forecasts for ARIMA-XGBoost hybrid."""
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
    predictions = []

    # Map ARIMA features
    n_total = len(combined)
    arima_feat = np.full(n_total, np.nan)
    arima_feat[:n_train] = arima_fitted
    arima_feat[n_train:] = arima_forecast

    for i in range(n_test):
        idx = n_train + i
        sub_df = combined.iloc[:idx + 1].copy()
        sub_df_feat = engineer_features(sub_df, cfg, commodity="KVPI")

        X_curr = sub_df_feat.iloc[-1:].reindex(columns=feature_cols, fill_value=0)
        X_curr["arima_fitted"] = arima_feat[idx]

        pred_val = model.predict(X_curr)[0]
        predictions.append(pred_val)
        combined.loc[idx, target] = pred_val

    return np.array(predictions)


def fit_arima_xgb_hybrid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_names: List[str],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Fit a hybrid ARIMA + XGBoost model.
    """
    from src.models.statistical import fit_auto_arima, predict_arima
    from src.models.ml_models import train_xgboost, get_xgb_feature_importance
    from src.evaluation import compute_all_metrics
    import xgboost as xgb

    target = cfg["preprocessing"]["target_column"]
    metrics_list = cfg.get("evaluation", {}).get("metrics", ["RMSE", "MAE", "MAPE", "sMAPE"])

    numeric_features = [
        f for f in feature_names
        if f in train_df.columns and train_df[f].dtype in [np.float64, np.int64, np.float32, np.int32, np.bool_]
    ]

    logger.info("═" * 50)
    logger.info("  ARIMA–XGBOOST HYBRID MODEL")
    logger.info("═" * 50)

    train_y = train_df[target].values
    test_y = test_df[target].values
    horizon = len(test_y)

    # Step 1: Fit ARIMA
    logger.info("Step 1/3: Fitting ARIMA component …")
    arima_model = fit_auto_arima(train_y, cfg)

    # ARIMA in-sample fitted values
    arima_fitted = train_y - arima_model.resid()

    # ARIMA out-of-sample forecast
    arima_output = predict_arima(arima_model, horizon, return_ci=False)
    arima_forecast = arima_output["predictions"]

    # Step 2: Augment features
    logger.info("Step 2/3: Augmenting features with ARIMA component …")

    # Align lengths
    n_train = len(train_df)
    arima_feat_train = np.full(n_train, np.nan)
    arima_feat_train[-len(arima_fitted):] = arima_fitted

    train_augmented = train_df[numeric_features].copy()
    train_augmented["arima_fitted"] = arima_feat_train

    test_augmented = test_df[numeric_features].copy()
    arima_feat_test = np.full(len(test_df), np.nan)
    arima_feat_test[:len(arima_forecast)] = arima_forecast
    test_augmented["arima_fitted"] = arima_feat_test

    # Drop NaN rows
    valid_mask_train = ~train_augmented.isna().any(axis=1)
    X_train = train_augmented[valid_mask_train]
    y_train = train_df.loc[valid_mask_train.values, target]

    valid_mask_test = ~test_augmented.isna().any(axis=1)
    X_test = test_augmented[valid_mask_test]
    y_test_valid = test_df.loc[valid_mask_test.values, target]

    augmented_features = list(X_train.columns)

    # Step 3: Train XGBoost on augmented features
    logger.info("Step 3/3: Training XGBoost on augmented features …")

    xgb_model, xgb_params = train_xgboost(X_train, y_train, cfg, seed)

    strategy = cfg["evaluation"].get("strategy", "recursive")
    if strategy == "recursive":
        logger.info("Generating recursive out-of-sample forecast for ARIMA-XGBoost…")
        xgb_pred = predict_recursive_arima_xgb(
            xgb_model, train_df, test_df, numeric_features, arima_forecast, arima_fitted, cfg
        )
        # Recursive forecaster fills all values step-by-step, so use
        # the full test target for evaluation (not the NaN-masked subset)
        y_test_eval = test_df[target].values
    else:
        logger.info("Generating one-step-ahead rolling forecast for ARIMA-XGBoost…")
        xgb_pred = xgb_model.predict(X_test)
        y_test_eval = y_test_valid.values

    metrics = compute_all_metrics(y_test_eval, xgb_pred, metrics_list)
    importance = get_xgb_feature_importance(xgb_model, augmented_features)

    logger.info(f"ARIMA–XGBoost Hybrid — RMSE: {metrics.get('RMSE', 'N/A'):.4f}")

    return {
        "predictions": xgb_pred,
        "y_test": y_test_eval,
        "metrics": metrics,
        "arima_model": arima_model,
        "xgb_model": xgb_model,
        "feature_importance": importance,
        "params": xgb_params,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. UNIFIED HYBRID RUNNER
# ═════════════════════════════════════════════════════════════════════════════


def run_hybrid_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_y: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
    feature_names: List[str],
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all configured hybrid models.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Train/test DataFrames with features.
    train_y, test_y : np.ndarray
        Train/test target arrays.
    cfg : dict
        Pipeline configuration.
    feature_names : list of str
        Feature column names.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Model name → results dict.
    """
    hybrid_cfg = cfg.get("models", {}).get("hybrid", {})
    results = {}

    # ARIMA-LSTM
    if hybrid_cfg.get("arima_lstm", {}).get("enabled", True):
        try:
            result = fit_arima_lstm_hybrid(train_y, test_y, cfg, seed)
            results["ARIMA_LSTM"] = result
        except Exception as e:
            logger.error(f"ARIMA-LSTM hybrid failed: {e}", exc_info=True)
        gc.collect()

    # ARIMA-XGBoost
    if hybrid_cfg.get("arima_xgb", {}).get("enabled", True):
        try:
            result = fit_arima_xgb_hybrid(
                train_df, test_df, cfg, feature_names, seed
            )
            results["ARIMA_XGBoost"] = result
        except Exception as e:
            logger.error(f"ARIMA-XGBoost hybrid failed: {e}", exc_info=True)
        gc.collect()

    return results
