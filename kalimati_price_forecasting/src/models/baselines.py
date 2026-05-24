"""
baselines.py — Naïve Forecasting Baselines
============================================

Implements simple forecasting methods that serve as lower-bound benchmarks:

    1. **Naïve Forecast**: Predict last known value (random walk).
    2. **Seasonal Naïve**: Predict the value from the same day of the
       previous seasonal cycle (default: weekly).

These baselines are essential for establishing whether more complex models
provide genuine predictive value (Hyndman & Athanasopoulos, 2021).

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("kalimati.models.baselines")


class NaiveForecast:
    """
    Naïve (random-walk) forecaster.

    Predicts the last observed value for all future time steps.
    Equivalent to a no-change forecast: ŷ_{t+h} = y_t for all h.
    """

    def __init__(self):
        self.last_value: Optional[float] = None
        self.name = "Naive"

    def fit(self, y: np.ndarray) -> "NaiveForecast":
        """
        Store the last observed value.

        Parameters
        ----------
        y : array-like
            Training time series.

        Returns
        -------
        self
        """
        y = np.asarray(y)
        self.last_value = float(y[-1])
        logger.debug(f"Naive fitted. Last value: {self.last_value:.2f}")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate flat forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Array of length ``horizon`` filled with the last value.
        """
        if self.last_value is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return np.full(horizon, self.last_value)


class SeasonalNaiveForecast:
    """
    Seasonal naïve forecaster.

    Predicts the value from the same position in the previous seasonal
    cycle: ŷ_{t+h} = y_{t+h-m} where m is the season length.
    """

    def __init__(self, season_length: int = 7):
        """
        Parameters
        ----------
        season_length : int
            Number of periods in one seasonal cycle (default: 7 for weekly).
        """
        self.season_length = season_length
        self.last_season: Optional[np.ndarray] = None
        self.name = f"Seasonal_Naive_{season_length}"

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecast":
        """
        Store the last full seasonal cycle.

        Parameters
        ----------
        y : array-like
            Training time series.

        Returns
        -------
        self
        """
        y = np.asarray(y)
        m = self.season_length
        if len(y) < m:
            logger.warning(
                f"Training data ({len(y)}) shorter than season length ({m}). "
                f"Falling back to naive."
            )
            self.last_season = np.full(m, y[-1])
        else:
            self.last_season = y[-m:]
        logger.debug(f"Seasonal Naive fitted (m={m})")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate seasonal forecasts by repeating the last cycle.

        Parameters
        ----------
        horizon : int
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Forecasts of length ``horizon``.
        """
        if self.last_season is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        m = self.season_length
        n_repeats = (horizon // m) + 1
        tiled = np.tile(self.last_season, n_repeats)
        return tiled[:horizon]


def run_baselines(
    train_y: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Run all baseline models and compute forecasts + metrics.

    Predictions are always of length ``len(test_y)`` so that
    ``run_all.py`` can evaluate at any horizon by slicing.

    Parameters
    ----------
    train_y : array-like
        Training target values.
    test_y : array-like
        Test target values.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Model name → {predictions, metrics, model}.
    """
    from src.evaluation import compute_all_metrics

    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    h = len(test_y)  # Always forecast the full test set
    metrics_list = cfg.get("evaluation", {}).get("metrics", ["RMSE", "MAE", "MAPE", "sMAPE"])
    results = {}

    # Naïve
    if cfg.get("models", {}).get("baselines", {}).get("naive", True):
        naive = NaiveForecast()
        naive.fit(train_y)
        pred = naive.predict(h)
        metrics = compute_all_metrics(test_y, pred, metrics_list)
        results["Naive"] = {"predictions": pred, "metrics": metrics, "model": naive}
        logger.info(f"Naive forecast — RMSE: {metrics.get('RMSE', 'N/A'):.4f}")

    # Seasonal Naïve
    sn_cfg = cfg.get("models", {}).get("baselines", {}).get("seasonal_naive", {})
    if sn_cfg:
        m = sn_cfg.get("season_length", 7)
        sn = SeasonalNaiveForecast(season_length=m)
        sn.fit(train_y)
        pred = sn.predict(h)
        metrics = compute_all_metrics(test_y, pred, metrics_list)
        results[sn.name] = {"predictions": pred, "metrics": metrics, "model": sn}
        logger.info(f"Seasonal Naive (m={m}) — RMSE: {metrics.get('RMSE', 'N/A'):.4f}")

    return results
