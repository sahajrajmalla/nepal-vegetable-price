"""
evaluation.py — Time-Series Evaluation Framework
=================================================

Provides:
    • Forecasting metrics (RMSE, MAE, MAPE, sMAPE)
    • Time-series cross-validation (expanding window + rolling origin)
    • Multi-horizon evaluation (7, 14, 30, 90 days)
    • Prediction interval generation (bootstrap & quantile)
    • Residual diagnostics (ACF, Ljung-Box, normality)
    • Diebold-Mariano test for statistical significance
    • Summary table generation (Markdown + CSV)
    • Model ranking

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import TimeSeriesSplit

from src.utils import get_config_value

logger = logging.getLogger("kalimati.evaluation")


# ═════════════════════════════════════════════════════════════════════════════
# 1. METRICS
# ═════════════════════════════════════════════════════════════════════════════


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Handles zero values by adding a small epsilon.
    """
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).

    More robust than MAPE when actuals are near zero.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).

    Measures the proportion of variance in the target explained by
    the model.  R² = 1 − (SS_res / SS_tot).  A perfect model yields
    R² = 1;  a constant-mean predictor yields R² = 0.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute a suite of forecasting metrics.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    metrics : list of str, optional
        Metric names to compute. Default: all.

    Returns
    -------
    dict
        Metric name → value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        logger.warning("No valid observations for metric computation")
        return {}

    available = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
    }

    if metrics is None:
        metrics = list(available.keys())

    results = {}
    for m in metrics:
        if m in available:
            results[m] = available[m](y_true, y_pred)
        else:
            logger.warning(f"Unknown metric: {m}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. TIME-SERIES SPLITS
# ═════════════════════════════════════════════════════════════════════════════


def fixed_split(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data at a fixed date boundary defined in config.

    If the configured split date falls outside the data's date range,
    the function automatically falls back to an 80/20 chronological
    split and logs a warning.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset for one commodity.
    cfg : dict
        Pipeline configuration.
    date_col : str
        Date column name.

    Returns
    -------
    tuple
        (train_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    data_min = df[date_col].min()
    data_max = df[date_col].max()

    train_end = pd.Timestamp(cfg["evaluation"]["train_end_date"])
    test_start = pd.Timestamp(cfg["evaluation"]["test_start_date"])

    # ── Auto-adapt if configured dates are outside the data range ──
    if train_end >= data_max or test_start > data_max:
        split_idx = int(len(df) * 0.80)
        train_end = df.iloc[split_idx][date_col]
        test_start = df.iloc[split_idx + 1][date_col]
        logger.warning(
            f"Configured split dates exceed data range "
            f"({data_min.date()} → {data_max.date()}). "
            f"Auto-adapting to 80/20 split at {train_end.date()}."
        )

    train = df[df[date_col] <= train_end].copy()
    test = df[df[date_col] >= test_start].copy()

    logger.info(
        f"Train/test split: train {len(train):,} rows "
        f"({data_min.date()} → {train_end.date()}) | "
        f"test {len(test):,} rows "
        f"({test_start.date()} → {data_max.date()})"
    )
    return train, test


def rolling_origin_cv(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    horizon: int = 30,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate rolling-origin (expanding window) cross-validation indices.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame for one commodity (sorted by date).
    cfg : dict
        Pipeline configuration.
    horizon : int
        Forecast horizon (days) for the test fold.

    Returns
    -------
    list of (train_idx, test_idx) tuples
        Each element contains NumPy arrays of integer indices.
    """
    cv_cfg = cfg["evaluation"]["cv"]
    n = len(df)
    min_train = cv_cfg.get("min_train_size", 730)
    step = cv_cfg.get("step_size", 60)
    gap = cv_cfg.get("gap", 0)
    n_splits = cv_cfg.get("n_splits", 5)

    folds = []
    start = min_train

    while start + gap + horizon <= n and len(folds) < n_splits:
        train_idx = np.arange(0, start)
        test_start = start + gap
        test_end = min(test_start + horizon, n)
        test_idx = np.arange(test_start, test_end)

        folds.append((train_idx, test_idx))
        start += step

    logger.info(f"Rolling-origin CV: {len(folds)} folds (horizon={horizon})")
    return folds


# ═════════════════════════════════════════════════════════════════════════════
# 3. MULTI-HORIZON EVALUATION
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_horizons(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int] = None,
    metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy at multiple horizons.

    Parameters
    ----------
    y_true : array-like
        Full test-set actual values.
    y_pred : array-like
        Full test-set predicted values.
    horizons : list of int
        Horizons to evaluate (e.g., [7, 14, 30, 90]).
    metrics : list of str
        Metrics to compute.

    Returns
    -------
    pd.DataFrame
        Rows = horizons, columns = metrics.
    """
    if horizons is None:
        horizons = [7, 14, 30, 90]
    if metrics is None:
        metrics = ["RMSE", "MAE", "MAPE", "sMAPE"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = []
    for h in horizons:
        n = min(h, len(y_true))
        if n == 0:
            continue
        m = compute_all_metrics(y_true[:n], y_pred[:n], metrics)
        m["Horizon"] = h
        m["N_Samples"] = n
        results.append(m)

    return pd.DataFrame(results).set_index("Horizon")


# ═════════════════════════════════════════════════════════════════════════════
# 4. RESIDUAL DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════


def residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_lags: int = 20,
) -> Dict[str, Any]:
    """
    Compute residual diagnostic statistics.

    Returns
    -------
    dict
        'residuals': residual array
        'mean': mean of residuals (should be ≈ 0)
        'std': std of residuals
        'skewness': skewness (should be ≈ 0)
        'kurtosis': excess kurtosis
        'ljung_box_stat': Ljung-Box Q statistic
        'ljung_box_pvalue': Ljung-Box p-value (H₀: no autocorrelation)
        'shapiro_stat': Shapiro-Wilk statistic
        'shapiro_pvalue': Shapiro-Wilk p-value (H₀: normality)
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    residuals = residuals[~np.isnan(residuals)]

    result = {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "skewness": float(sp_stats.skew(residuals)),
        "kurtosis": float(sp_stats.kurtosis(residuals)),
    }

    # Ljung-Box test
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(residuals, lags=[n_lags], return_df=True)
        result["ljung_box_stat"] = float(lb.iloc[0]["lb_stat"])
        result["ljung_box_pvalue"] = float(lb.iloc[0]["lb_pvalue"])
    except Exception as e:
        logger.debug(f"Ljung-Box test failed: {e}")
        result["ljung_box_stat"] = np.nan
        result["ljung_box_pvalue"] = np.nan

    # Shapiro-Wilk (limited to 5000 samples)
    try:
        sample = residuals[:5000]
        if len(sample) >= 20:
            stat, p = sp_stats.shapiro(sample)
            result["shapiro_stat"] = float(stat)
            result["shapiro_pvalue"] = float(p)
        else:
            result["shapiro_stat"] = np.nan
            result["shapiro_pvalue"] = np.nan
    except Exception as e:
        logger.debug(f"Shapiro-Wilk test failed: {e}")
        result["shapiro_stat"] = np.nan
        result["shapiro_pvalue"] = np.nan

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. DIEBOLD-MARIANO TEST
# ═════════════════════════════════════════════════════════════════════════════


def diebold_mariano_test(
    y_true: np.ndarray,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
) -> Dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H₀: The two forecasts have equal accuracy.
    H₁: They differ in accuracy.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    pred_1 : array-like
        Predictions from model 1.
    pred_2 : array-like
        Predictions from model 2.
    horizon : int
        Forecast horizon (for HAC adjustment).
    loss : str
        Loss function: 'squared' (MSE) or 'absolute' (MAE).

    Returns
    -------
    dict
        'dm_stat': DM test statistic
        'dm_pvalue': two-sided p-value
    """
    y_true = np.asarray(y_true)
    e1 = y_true - np.asarray(pred_1)
    e2 = y_true - np.asarray(pred_2)

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_mean = np.mean(d)

    # HAC variance (Newey-West with bandwidth = horizon - 1)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, horizon):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * gamma_k

    variance = (gamma_0 + gamma_sum) / n

    if variance <= 0:
        return {"dm_stat": np.nan, "dm_pvalue": np.nan}

    dm_stat = d_mean / np.sqrt(variance)
    dm_pvalue = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))

    return {"dm_stat": float(dm_stat), "dm_pvalue": float(dm_pvalue)}


# ═════════════════════════════════════════════════════════════════════════════
# 6. PREDICTION INTERVALS
# ═════════════════════════════════════════════════════════════════════════════


def bootstrap_prediction_intervals(
    y_train_resid: np.ndarray,
    point_forecast: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate prediction intervals via residual bootstrapping.

    Parameters
    ----------
    y_train_resid : array-like
        In-sample residuals from the fitted model.
    point_forecast : array-like
        Out-of-sample point forecasts.
    n_bootstrap : int
        Number of bootstrap replications.
    confidence : float
        Confidence level (e.g., 0.95 for 95% PI).
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (lower_bound, upper_bound) arrays.
    """
    rng = np.random.RandomState(seed)
    resid = np.asarray(y_train_resid)
    forecast = np.asarray(point_forecast)
    n_out = len(forecast)

    # Bootstrap samples of residuals
    boot_forecasts = np.zeros((n_bootstrap, n_out))
    for i in range(n_bootstrap):
        sampled_resid = rng.choice(resid, size=n_out, replace=True)
        boot_forecasts[i] = forecast + sampled_resid

    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_forecasts, alpha * 100, axis=0)
    upper = np.percentile(boot_forecasts, (1 - alpha) * 100, axis=0)

    return lower, upper


# ═════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY TABLE GENERATION
# ═════════════════════════════════════════════════════════════════════════════


def generate_results_table(
    results: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compile model comparison results into a summary table.

    Parameters
    ----------
    results : list of dict
        Each dict: {Model, Commodity, Horizon, RMSE, MAE, MAPE, sMAPE, ...}.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table sorted by RMSE.
    """
    df = pd.DataFrame(results)

    # Round numeric columns
    numeric_cols = ["RMSE", "MAE", "MAPE", "sMAPE"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Sort by commodity, horizon, RMSE
    sort_cols = [c for c in ["Commodity", "Horizon", "RMSE"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def rank_models(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """
    Rank models per commodity and horizon.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from ``generate_results_table``.
    metric : str
        Metric to rank by (lower is better).

    Returns
    -------
    pd.DataFrame
        Results table with an added ``Rank`` column.
    """
    df = results_df.copy()
    group_cols = [c for c in ["Commodity", "Horizon"] if c in df.columns]

    if group_cols:
        df["Rank"] = df.groupby(group_cols)[metric].rank(method="min").astype(int)
    else:
        df["Rank"] = df[metric].rank(method="min").astype(int)

    return df.sort_values(group_cols + ["Rank"]).reset_index(drop=True)


def save_results(
    results_df: pd.DataFrame,
    cfg: Dict[str, Any],
    filename: str = "model_comparison",
) -> None:
    """
    Save results table as CSV and Markdown.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table.
    cfg : dict
        Pipeline configuration.
    filename : str
        Base filename (without extension).
    """
    out_dir = Path(cfg["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / f"{filename}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results CSV saved: {csv_path}")

    # Markdown
    md_path = out_dir / f"{filename}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Model Comparison Results\n\n")
        f.write(f"Generated by Kalimati Price Forecasting Pipeline\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n")
    logger.info(f"Results Markdown saved: {md_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. BEST MODEL SELECTOR
# ═════════════════════════════════════════════════════════════════════════════


def select_best_model(
    results_df: pd.DataFrame,
    commodity: str,
    horizon: int,
    metric: str = "RMSE",
) -> str:
    """
    Select the best model for a given commodity and horizon.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results table.
    commodity : str
        Commodity name.
    horizon : int
        Forecast horizon.
    metric : str
        Selection metric (lower is better).

    Returns
    -------
    str
        Name of the best model.
    """
    mask = (results_df["Commodity"] == commodity) & (results_df["Horizon"] == horizon)
    subset = results_df[mask]

    if subset.empty:
        logger.warning(f"No results for {commodity}, horizon={horizon}")
        return "Unknown"

    best_idx = subset[metric].idxmin()
    best_model = subset.loc[best_idx, "Model"]
    best_val = subset.loc[best_idx, metric]

    logger.info(
        f"Best model for {commodity} (h={horizon}): "
        f"{best_model} ({metric}={best_val:.4f})"
    )
    return best_model
