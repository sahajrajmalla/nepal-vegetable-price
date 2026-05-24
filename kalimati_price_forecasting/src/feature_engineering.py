"""
feature_engineering.py — Feature Construction for Time-Series Forecasting
=========================================================================

Constructs a rich feature matrix from cleaned daily price data:

    • Autoregressive lags (1, 2, 3, 7, 14, 21, 30 days)
    • Rolling statistics (mean, std, min, max, median over 7/14/30 windows)
    • Exponentially Weighted Moving Averages (EWMA)
    • First-order and seasonal differences
    • Calendar features (day-of-week, month, sin/cos day-of-year, weekend)
    • Festival dummy variables (Dashain, Tihar, Chhath, Holi, Teej, New Year)
    • Price spread (Max − Min) and velocity (rate of change)
    • Rolling coefficient of variation (volatility proxy)

All features are computed per-commodity to prevent cross-contamination.

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_config_value, sanitize_commodity_name

logger = logging.getLogger("kalimati.features")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lag Features
# ─────────────────────────────────────────────────────────────────────────────


def add_lag_features(
    df: pd.DataFrame,
    target: str = "Average",
    lags: List[int] = None,
) -> pd.DataFrame:
    """
    Create lagged versions of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Single-commodity daily data sorted by date.
    target : str
        Column name of the target variable.
    lags : list of int
        Lag periods (e.g., [1, 7, 14, 30]).

    Returns
    -------
    pd.DataFrame
        DataFrame with added lag columns (e.g., ``Average_lag_7``).
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 30]

    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling Statistics
# ─────────────────────────────────────────────────────────────────────────────


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "Average",
    windows: List[int] = None,
    stats: List[str] = None,
) -> pd.DataFrame:
    """
    Compute rolling window statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Single-commodity daily data sorted by date.
    target : str
        Column name of the target variable.
    windows : list of int
        Rolling window sizes in days.
    stats : list of str
        Statistics to compute: 'mean', 'std', 'min', 'max', 'median'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling feature columns.
    """
    if windows is None:
        windows = [7, 14, 30]
    if stats is None:
        stats = ["mean", "std", "min", "max", "median"]

    for window in windows:
        rolling = df[target].rolling(window=window, min_periods=window)
        for stat in stats:
            col_name = f"{target}_roll_{stat}_{window}"
            if stat == "mean":
                df[col_name] = rolling.mean().shift(1)
            elif stat == "std":
                df[col_name] = rolling.std().shift(1)
            elif stat == "min":
                df[col_name] = rolling.min().shift(1)
            elif stat == "max":
                df[col_name] = rolling.max().shift(1)
            elif stat == "median":
                df[col_name] = rolling.median().shift(1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. EWMA Features
# ─────────────────────────────────────────────────────────────────────────────


def add_ewma_features(
    df: pd.DataFrame,
    target: str = "Average",
    spans: List[int] = None,
) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Moving Averages.

    Parameters
    ----------
    df : pd.DataFrame
        Single-commodity daily data sorted by date.
    target : str
        Target column.
    spans : list of int
        EWMA span parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with EWMA columns.
    """
    if spans is None:
        spans = [7, 14, 30]

    for span in spans:
        df[f"{target}_ewma_{span}"] = df[target].ewm(span=span, adjust=False).mean().shift(1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Differencing
# ─────────────────────────────────────────────────────────────────────────────


def add_differencing_features(
    df: pd.DataFrame,
    target: str = "Average",
    orders: List[int] = None,
) -> pd.DataFrame:
    """
    Compute first-order and seasonal differences.

    Parameters
    ----------
    df : pd.DataFrame
        Single-commodity daily data sorted by date.
    target : str
        Target column.
    orders : list of int
        Differencing periods (e.g., [1, 7] for daily and weekly differences).

    Returns
    -------
    pd.DataFrame
        DataFrame with difference columns (e.g., ``Average_diff_7``).
    """
    if orders is None:
        orders = [1, 7]

    for order in orders:
        df[f"{target}_diff_{order}"] = df[target].diff(order).shift(1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Calendar Features
# ─────────────────────────────────────────────────────────────────────────────


def add_calendar_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Extract calendar and cyclical temporal features from the Date column.

    Features
    --------
    - dayofweek (0=Mon, 6=Sun)
    - month (1-12)
    - dayofyear (1-366)
    - is_weekend (bool)
    - quarter (1-4)
    - week_of_year (1-53)
    - sin/cos encoded day-of-year for cyclical representation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Date`` column.
    cfg : dict
        Pipeline configuration (calendar feature toggles).

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar features added.
    """
    cal_cfg = get_config_value(cfg, "features", "calendar", default={})

    if cal_cfg.get("dayofweek", True):
        df["dayofweek"] = df["Date"].dt.dayofweek

    if cal_cfg.get("month", True):
        df["month"] = df["Date"].dt.month

    if cal_cfg.get("dayofyear", True):
        df["dayofyear"] = df["Date"].dt.dayofyear

    if cal_cfg.get("is_weekend", True):
        # Nepal uses Saturday as holiday; Friday is half-day in many contexts
        # Saturday (5) is the primary holiday
        df["is_weekend"] = df["Date"].dt.dayofweek.isin([5]).astype(int)

    if cal_cfg.get("quarter", True):
        df["quarter"] = df["Date"].dt.quarter

    if cal_cfg.get("week_of_year", True):
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

    if cal_cfg.get("sin_cos_encoding", True):
        day_of_year = df["Date"].dt.dayofyear
        df["sin_dayofyear"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df["cos_dayofyear"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Also encode day-of-week cyclically
        dow = df["Date"].dt.dayofweek
        df["sin_dayofweek"] = np.sin(2 * np.pi * dow / 7)
        df["cos_dayofweek"] = np.cos(2 * np.pi * dow / 7)

        # Encode month cyclically
        month = df["Date"].dt.month
        df["sin_month"] = np.sin(2 * np.pi * month / 12)
        df["cos_month"] = np.cos(2 * np.pi * month / 12)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Festival Features
# ─────────────────────────────────────────────────────────────────────────────


def _build_festival_ranges(cfg: Dict[str, Any]) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Build expanded festival date ranges (including lead/lag days).

    Returns
    -------
    dict
        festival_name → list of (start, end) Timestamp tuples.
    """
    festivals_cfg = cfg.get("festivals", {})
    ranges: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}

    for festival_name, fest_info in festivals_cfg.items():
        if not isinstance(fest_info, dict):
            continue

        lead = pd.Timedelta(days=fest_info.get("lead_days", 0))
        lag = pd.Timedelta(days=fest_info.get("lag_days", 0))
        date_pairs = fest_info.get("approximate_dates", [])

        expanded = []
        for pair in date_pairs:
            start = pd.Timestamp(pair[0]) - lead
            end = pd.Timestamp(pair[1]) + lag
            expanded.append((start, end))

        ranges[festival_name] = expanded

    return ranges


def add_festival_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add binary dummy variables for major Nepali festivals.

    Each festival gets:
      • ``fest_{name}``: 1 during the festival + lead/lag window, else 0.
      • ``fest_any``: 1 if any festival is active.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Date`` column.
    cfg : dict
        Pipeline configuration (festival definitions).

    Returns
    -------
    pd.DataFrame
        DataFrame with festival dummy columns.
    """
    festival_ranges = _build_festival_ranges(cfg)

    for fest_name, ranges in festival_ranges.items():
        col = f"fest_{fest_name}"
        df[col] = 0
        for start, end in ranges:
            mask = (df["Date"] >= start) & (df["Date"] <= end)
            df.loc[mask, col] = 1

    # Composite "any festival" flag
    fest_cols = [c for c in df.columns if c.startswith("fest_")]
    if fest_cols:
        df["fest_any"] = (df[fest_cols].sum(axis=1) > 0).astype(int)

    n_festival_days = df["fest_any"].sum() if "fest_any" in df.columns else 0
    logger.info(f"Festival features: {len(festival_ranges)} festivals, "
                f"{n_festival_days:,} festival-affected days")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Price-Derived Features
# ─────────────────────────────────────────────────────────────────────────────


def add_price_derived_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute features derived from price data.

    Features
    --------
    - ``spread``: Max − Min (intraday price range).
    - ``price_velocity``: Rate of change = (price − price_lag1) / price_lag1.
    - ``volatility_7``, ``volatility_14``, ``volatility_30``:
      Rolling coefficient of variation (std/mean).
    - ``price_momentum_7``: 7-day return (price / price_lag7 − 1).
    - ``price_acceleration``: Second difference (velocity change).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price columns.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with price-derived features.
    """
    target = cfg["preprocessing"]["target_column"]

    if get_config_value(cfg, "features", "spread", default=True):
        df["spread"] = (df["Maximum"] - df["Minimum"]).shift(1)

    if get_config_value(cfg, "features", "price_velocity", default=True):
        velocity = df[target].pct_change(periods=1)
        momentum = df[target].pct_change(periods=7)
        acceleration = velocity.diff(1)
        df["price_velocity"] = velocity.shift(1)
        df["price_momentum_7"] = momentum.shift(1)
        df["price_acceleration"] = acceleration.shift(1)

    if get_config_value(cfg, "features", "volatility", default=True):
        for window in [7, 14, 30]:
            rolling_mean = df[target].rolling(window=window, min_periods=window).mean()
            rolling_std = df[target].rolling(window=window, min_periods=window).std()
            df[f"volatility_{window}"] = (rolling_std / (rolling_mean + 1e-8)).shift(1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Master Feature Engineering Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def engineer_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    commodity: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a single commodity's data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily data for one commodity.
    cfg : dict
        Pipeline configuration.
    commodity : str, optional
        Commodity name (for logging).

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame (NaN rows from lagging will be present;
        the caller should handle trimming).
    """
    name = commodity or "Unknown"
    logger.info(f"Engineering features for: {name}")

    df = df.sort_values("Date").reset_index(drop=True)
    target = cfg["preprocessing"]["target_column"]

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Lag features
    lags = get_config_value(cfg, "features", "lags", default=[1, 7, 14, 30])
    df = add_lag_features(df, target=target, lags=lags)

    # Rolling features
    windows = get_config_value(cfg, "features", "rolling_windows", default=[7, 14, 30])
    stats = get_config_value(cfg, "features", "rolling_stats",
                              default=["mean", "std", "min", "max", "median"])
    df = add_rolling_features(df, target=target, windows=windows, stats=stats)

    # EWMA
    spans = get_config_value(cfg, "features", "ewma_spans", default=[7, 14, 30])
    df = add_ewma_features(df, target=target, spans=spans)

    # Differencing
    orders = get_config_value(cfg, "features", "differencing_orders", default=[1, 7])
    df = add_differencing_features(df, target=target, orders=orders)

    # Calendar
    df = add_calendar_features(df, cfg)

    # Festival dummies
    df = add_festival_features(df, cfg)

    # Price-derived
    df = add_price_derived_features(df, cfg)

    # Count features
    n_features = len([c for c in df.columns if c not in
                      ["Date", "Commodity", "Unit", "is_outlier", "is_imputed"]])
    logger.info(f"  → {n_features} features created for {name}")

    return df


def engineer_all_commodities(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Apply feature engineering to all commodities in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily data (all commodities).
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Commodity name → feature-enriched DataFrame.
    """
    result = {}
    commodities = sorted(df["Commodity"].unique())

    for commodity in commodities:
        sub = df[df["Commodity"] == commodity].copy()
        featured = engineer_features(sub, cfg, commodity=commodity)
        result[commodity] = featured

    logger.info(f"Feature engineering complete for {len(result)} commodities")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. Feature Selection Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature column names (excluding metadata & target).

    Automatically excludes:
    - Metadata columns (Date, Commodity, Unit, etc.)
    - Category/group string columns (Commodity_Category, Commodity_Group)
    - Any column with non-numeric dtype (object, string, category)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.

    Returns
    -------
    list of str
        Feature column names suitable for model input (all numeric).
    """
    exclude = {
        "Date", "Commodity", "Unit", "Minimum", "Maximum", "Average",
        "is_outlier", "is_imputed",
        "Commodity_Category", "Commodity_Group",
        "KVPI", "KVPI_Median", "N_Commodities", "Avg_Price", "Min_Price", "Max_Price",
    }
    numeric_dtypes = {"int64", "float64", "int32", "float32", "bool", "int8", "uint8"}
    return [
        c for c in df.columns
        if c not in exclude and str(df[c].dtype) in numeric_dtypes
    ]


def prepare_ml_dataset(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    target: str = "Average",
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare X (features) and y (target) arrays for ML models.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame for a single commodity.
    cfg : dict
        Pipeline configuration.
    target : str
        Target column name.
    drop_na : bool
        If True, drop rows with any NaN in features.

    Returns
    -------
    tuple
        (X, y, feature_names) where X is the feature matrix, y is the target,
        and feature_names lists the column names.
    """
    feature_cols = get_feature_columns(df)
    df_clean = df[["Date", target] + feature_cols].copy()

    if drop_na:
        df_clean = df_clean.dropna()

    X = df_clean[feature_cols]
    y = df_clean[target]
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import load_config, setup_logger

    cfg = load_config()
    setup_logger(
        "kalimati",
        log_file=cfg["logging"]["log_file"],
        level=cfg["logging"]["level"],
    )
    setup_logger("kalimati.features", level=cfg["logging"]["level"])

    # Load cleaned data
    cleaned_path = Path(cfg["data"]["cleaned_dir"]) / "kalimati_kvpi.csv"
    if not cleaned_path.exists():
        print("Run 02_preprocess.py first!")
        sys.exit(1)

    df = pd.read_csv(cleaned_path)
    df["Date"] = pd.to_datetime(df["Date"])
    commodity_features = engineer_all_commodities(df, cfg)

    for name, feat_df in commodity_features.items():
        print(f"{name}: {feat_df.shape}")
