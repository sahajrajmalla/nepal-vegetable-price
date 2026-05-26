"""
sota_models.py — State-of-the-Art Deep Learning Models via NeuralForecast
========================================================================

Implements modern Transformer and MLP-based models:
    1. PatchTST
    2. NBEATSx

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST, NBEATSx
except ImportError:
    NeuralForecast = None

logger = logging.getLogger("kalimati.models.sota")

def prepare_nf_data(
    df: pd.DataFrame,
    target: str = "Average",
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Prepare dataframe for NeuralForecast (requires ds, unique_id, y)."""
    cols = ["Date", "Commodity", target]
    if feature_cols:
        cols += feature_cols
    nf_df = df[cols].copy()
    nf_df = nf_df.rename(columns={
        "Date": "ds",
        "Commodity": "unique_id",
        target: "y"
    })
    return nf_df

def train_sota_models(
    train_df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
) -> Any:
    """
    Train PatchTST and NBEATSx models.
    """
    if NeuralForecast is None:
        raise ImportError("neuralforecast is not installed. Please pip install neuralforecast.")

    target = cfg["preprocessing"]["target_column"]
    nf_train = prepare_nf_data(train_df, target, feature_cols)

    # Use max horizon as h
    horizons = cfg["evaluation"]["horizons"]
    h = max(horizons)

    seq_length = cfg.get("models", {}).get("sota", {}).get("seq_length", 60)
    max_steps = cfg.get("models", {}).get("sota", {}).get("max_steps", 200)

    models = []
    
    # PatchTST
    patchtst = PatchTST(
        h=h,
        input_size=seq_length,
        patch_len=14,
        stride=14,
        max_steps=max_steps,
        scaler_type="standard",
        random_seed=42,
    )
    models.append(patchtst)

    # NBEATSx
    # Exclude non-numeric/high-cardinality from exogenous if any
    if feature_cols:
        futr_exog = [c for c in feature_cols if c not in ["unique_id", "ds", "y"]]
    else:
        futr_exog = None

    nbeatsx = NBEATSx(
        h=h,
        input_size=seq_length,
        futr_exog_list=futr_exog,
        hist_exog_list=None,
        max_steps=max_steps,
        scaler_type="standard",
        random_seed=42,
    )
    models.append(nbeatsx)

    nf = NeuralForecast(models=models, freq="D")
    logger.info(f"Training NeuralForecast models with h={h}, input_size={seq_length}")
    nf.fit(df=nf_train)

    return nf

def predict_sota(
    nf: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Predict using NeuralForecast models natively.
    """
    target = cfg["preprocessing"]["target_column"]
    
    # nf.predict() automatically predicts the next `h` steps.
    # To predict over the test set, we can use predict() directly if test set length == h.
    # If we need rolling forecasts or futr_df:
    nf_test = prepare_nf_data(test_df, target, feature_cols)
    
    # NeuralForecast uses futr_df for future exogenous variables (NBEATSx needs this)
    futr_df = nf_test.drop(columns=["y"]) if feature_cols else None
    
    preds = nf.predict(futr_df=futr_df)
    
    # preds contains unique_id, ds, and columns for each model
    return preds
