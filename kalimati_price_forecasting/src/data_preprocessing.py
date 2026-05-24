"""
data_preprocessing.py — Data Ingestion, Cleaning & KVPI Index Creation
========================================================================

Handles the full data pipeline from raw CSVs to the Kalimati Vegetable
Price Index (KVPI):

    1. Ingest & concatenate multiple raw CSV files (heterogeneous schemas)
    2. Schema normalisation (column names, date parsing, unit harmonisation)
    3. Duplicate detection & removal
    4. Outlier detection (IQR + domain constraints) with transparent logging
    5. Intelligent imputation (forward-fill → linear interpolation)
    6. KVPI construction (base-period normalized cross-sectional average)
    7. CSV export

Design rationale
----------------
* Every cleaning step is logged so the research paper can cite exact counts
  of removed/imputed records.
* The module is idempotent: re-running it on the same raw data produces
  identical outputs (deterministic).
* The KVPI is constructed from ALL commodities (≥365 days of data),
  providing a robust composite index that smooths individual fluctuations.

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    ensure_dirs,
    get_selected_commodities,
    load_config,
    sanitize_commodity_name,
    setup_logger,
)

logger = logging.getLogger("kalimati.preprocessing")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ingestion
# ─────────────────────────────────────────────────────────────────────────────


def load_raw_csvs(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and concatenate all raw CSV files listed in the configuration.

    Handles schema differences between the two known Kalimati datasets:
      • 2013-2021 file: has ``SN`` column, header in row 0, dates YYYY-MM-DD.
      • 2021-2023 file: **no header row**, dates in M/D/YYYY, prices may
        have ``Rs `` prefix, units may be upper-case ``KG``.

    The function auto-detects whether a CSV has a header by inspecting
    the first line for known column names.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration (must contain ``data.raw_dir``, ``data.raw_files``).

    Returns
    -------
    pd.DataFrame
        Concatenated raw dataframe with unified schema:
        [Commodity, Date, Unit, Minimum, Maximum, Average].
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_files = cfg["data"]["raw_files"]
    frames: List[pd.DataFrame] = []

    EXPECTED_COLS = ["Commodity", "Date", "Unit", "Minimum", "Maximum", "Average"]

    for fname in raw_files:
        fpath = raw_dir / fname
        if not fpath.exists():
            logger.warning(f"Raw file not found, skipping: {fpath}")
            continue

        logger.info(f"Loading {fpath.name}  ({fpath.stat().st_size / 1e6:.1f} MB)")

        # ── Auto-detect whether file has a header row ──
        with open(fpath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        known_headers = {"SN", "Commodity", "Date", "Unit", "Minimum", "Maximum", "Average"}
        first_fields = {field.strip() for field in first_line.split(",")}
        has_header = bool(known_headers & first_fields)

        if has_header:
            df = pd.read_csv(fpath, encoding="utf-8")
        else:
            # Headerless file — assign column names manually
            df = pd.read_csv(
                fpath, encoding="utf-8", header=None,
                names=EXPECTED_COLS,
            )
            logger.info(f"  ↳ No header detected — assigned columns: {EXPECTED_COLS}")

        # Normalise column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Drop SN column if present
        if "SN" in df.columns:
            df = df.drop(columns=["SN"])

        # Ensure required columns exist
        required = set(EXPECTED_COLS)
        missing = required - set(df.columns)
        if missing:
            logger.error(f"File {fname} missing columns: {missing}. Skipping.")
            continue

        frames.append(df)
        logger.info(f"  → {len(df):,} records loaded from {fname}")

    if not frames:
        raise RuntimeError("No valid raw CSV files found. Check config paths.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined raw dataset: {len(combined):,} records")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema Normalisation
# ─────────────────────────────────────────────────────────────────────────────


def _clean_price_value(val: Any) -> float:
    """
    Parse a price cell that may contain:
      • A plain numeric string ("35.0")
      • A prefixed string ("Rs 45.00")
      • NaN or empty string

    Returns
    -------
    float
        Cleaned numeric value, or np.nan.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Remove currency prefix (case-insensitive)
    s = re.sub(r"(?i)^rs\.?\s*", "", s)
    # Remove commas
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonise column types and formats across heterogeneous CSV sources.

    Steps
    -----
    1. Clean price columns (remove ``Rs`` prefix, parse to float).
    2. Parse dates robustly (handles both YYYY-MM-DD and M/D/YYYY).
    3. Standardise ``Unit`` column to title-case ("Kg").
    4. Strip whitespace from ``Commodity`` names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw concatenated dataframe.

    Returns
    -------
    pd.DataFrame
        Schema-normalised dataframe sorted by [Commodity, Date].
    """
    logger.info("Normalising schema …")

    # --- Price columns ---
    for col in ["Minimum", "Maximum", "Average"]:
        df[col] = df[col].apply(_clean_price_value)

    # --- Date parsing ---
    # Try ISO format first, then US format
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)

    # --- Unit harmonisation ---
    df["Unit"] = df["Unit"].astype(str).str.strip().str.title()
    unit_counts = df["Unit"].value_counts()
    logger.info(f"Unit distribution:\n{unit_counts.to_string()}")

    # --- Commodity name cleanup ---
    df["Commodity"] = df["Commodity"].astype(str).str.strip()

    # Sort
    df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
    logger.info(f"Schema normalisation complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Duplicate Handling
# ─────────────────────────────────────────────────────────────────────────────


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact and near-duplicates.

    For the overlapping period (May 2021), both CSVs may contain records.
    We keep the record with the more recent/complete data (last occurrence).

    Parameters
    ----------
    df : pd.DataFrame
        Normalised dataframe.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    """
    n_before = len(df)

    # Exact duplicates
    df = df.drop_duplicates(subset=["Commodity", "Date"], keep="last")
    n_after = len(df)
    n_removed = n_before - n_after

    logger.info(
        f"Duplicate removal: {n_removed:,} duplicates dropped "
        f"({n_before:,} → {n_after:,})"
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Outlier Detection & Treatment
# ─────────────────────────────────────────────────────────────────────────────


def detect_outliers(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Flag and cap outliers using the IQR method combined with domain rules.

    Strategy
    --------
    1. Per-commodity IQR bounds on the ``Average`` price.
    2. Hard domain clamps: [min_price, max_price] from config.
    3. Flagged records are capped (winsorised), not dropped, to preserve
       temporal continuity.
    4. A boolean column ``is_outlier`` is added.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    cfg : dict
        Pipeline configuration (outlier thresholds).

    Returns
    -------
    pd.DataFrame
        Dataframe with outliers treated and flagged.
    """
    iqr_mult = cfg["preprocessing"]["outlier"]["iqr_multiplier"]
    price_min = cfg["preprocessing"]["outlier"]["min_price"]
    price_max = cfg["preprocessing"]["outlier"]["max_price"]
    target = cfg["preprocessing"]["target_column"]

    df["is_outlier"] = False
    n_outliers = 0

    for commodity in df["Commodity"].unique():
        mask = df["Commodity"] == commodity
        series = df.loc[mask, target]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = max(q1 - iqr_mult * iqr, price_min)
        upper = min(q3 + iqr_mult * iqr, price_max)

        outlier_mask = mask & ((df[target] < lower) | (df[target] > upper))
        count = outlier_mask.sum()
        n_outliers += count

        if count > 0:
            logger.debug(
                f"  {commodity}: {count} outliers detected "
                f"(bounds: [{lower:.1f}, {upper:.1f}])"
            )
            df.loc[outlier_mask, "is_outlier"] = True
            # Winsorise price columns (use per-commodity IQR bounds)
            for col in ["Minimum", "Maximum", "Average"]:
                df.loc[outlier_mask, col] = df.loc[outlier_mask, col].clip(
                    lower=lower, upper=upper
                )

    logger.info(f"Outlier detection: {n_outliers:,} values flagged and winsorised")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Missing Date Imputation
# ─────────────────────────────────────────────────────────────────────────────


def impute_missing_dates(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Ensure every commodity has a continuous daily index, then impute gaps.

    Steps
    -----
    1. For each commodity, reindex to a full daily date range.
    2. Forward-fill short gaps (≤ max_gap_days).
    3. Linear interpolation for remaining NaN.
    4. Flag imputed rows with ``is_imputed`` boolean.

    Parameters
    ----------
    df : pd.DataFrame
        Deduplicated, outlier-treated dataframe.
    cfg : dict
        Pipeline configuration (imputation settings).

    Returns
    -------
    pd.DataFrame
        Fully continuous daily dataframe with imputation flags.
    """
    max_gap = cfg["preprocessing"]["imputation"]["max_gap_days"]
    target = cfg["preprocessing"]["target_column"]
    price_cols = cfg["preprocessing"]["price_columns"]

    frames = []

    for commodity in sorted(df["Commodity"].unique()):
        sub = df[df["Commodity"] == commodity].copy()
        sub = sub.set_index("Date").sort_index()

        # Full daily date range
        full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
        sub = sub.reindex(full_idx)

        # Flag imputed rows
        sub["is_imputed"] = sub[target].isna()

        # Forward-fill then interpolate
        sub[price_cols] = sub[price_cols].ffill(limit=max_gap)
        sub[price_cols] = sub[price_cols].interpolate(
            method="linear", limit_direction="both"
        )

        # Fill metadata
        sub["Commodity"] = commodity
        sub["Unit"] = sub["Unit"].ffill().bfill()
        if "is_outlier" in sub.columns:
            sub["is_outlier"] = sub["is_outlier"].fillna(False)

        sub.index.name = "Date"
        sub = sub.reset_index()
        frames.append(sub)

    result = pd.concat(frames, ignore_index=True)
    n_imputed = result["is_imputed"].sum()
    logger.info(f"Missing date imputation: {n_imputed:,} daily records imputed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def create_aggregations(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Create daily, weekly, and monthly aggregate datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily dataframe.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Keys: 'daily', 'weekly', 'monthly'. Values: aggregated DataFrames.
    """
    price_cols = cfg["preprocessing"]["price_columns"]
    aggs = {"daily": df.copy()}

    agg_cfg = cfg["preprocessing"]["aggregation"]

    if agg_cfg.get("weekly", False):
        weekly = (
            df.groupby(["Commodity", pd.Grouper(key="Date", freq="W-MON")])
            .agg(
                {
                    "Minimum": "min",
                    "Maximum": "max",
                    "Average": "mean",
                    "Unit": "first",
                }
            )
            .reset_index()
        )
        aggs["weekly"] = weekly
        logger.info(f"Weekly aggregation: {len(weekly):,} records")

    if agg_cfg.get("monthly", False):
        monthly = (
            df.groupby(["Commodity", pd.Grouper(key="Date", freq="MS")])
            .agg(
                {
                    "Minimum": "min",
                    "Maximum": "max",
                    "Average": "mean",
                    "Unit": "first",
                }
            )
            .reset_index()
        )
        aggs["monthly"] = monthly
        logger.info(f"Monthly aggregation: {len(monthly):,} records")

    return aggs


# ─────────────────────────────────────────────────────────────────────────────
# 7. Export
# ─────────────────────────────────────────────────────────────────────────────


def save_cleaned_data(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    commodities: Optional[List[str]] = None,
) -> None:
    """
    Save cleaned data as Parquet files (full dataset + per-commodity splits).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily dataframe.
    cfg : dict
        Pipeline configuration.
    commodities : list of str, optional
        Subset of commodities to export. If None, exports all.
    """
    out_dir = Path(cfg["data"]["cleaned_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full dataset (CSV)
    full_path = out_dir / "kalimati_cleaned_full.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"Saved full cleaned dataset: {full_path}")

    # Per-commodity CSVs
    if commodities is None:
        commodities = df["Commodity"].unique().tolist()

    for commodity in commodities:
        sub = df[df["Commodity"] == commodity]
        if sub.empty:
            logger.warning(f"No data for commodity '{commodity}', skipping export.")
            continue
        slug = sanitize_commodity_name(commodity)
        csv_path = out_dir / f"{slug}.csv"
        sub.to_csv(csv_path, index=False)
        logger.debug(f"  Saved {commodity} → {csv_path}")

    logger.info(f"Exported {len(commodities)} commodity CSV files")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Data Quality Report
# ─────────────────────────────────────────────────────────────────────────────


def generate_data_quality_report(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate a per-commodity data quality summary.

    Columns
    -------
    Commodity, DateRange, TotalDays, RecordCount, MissingDays,
    MissingPct, OutlierCount, ImputedCount, PriceMean, PriceStd.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily dataframe.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Quality report dataframe.
    """
    target = cfg["preprocessing"]["target_column"]
    rows = []

    for commodity in sorted(df["Commodity"].unique()):
        sub = df[df["Commodity"] == commodity].copy()
        sub = sub.sort_values("Date")

        date_min = sub["Date"].min()
        date_max = sub["Date"].max()
        total_days = (date_max - date_min).days + 1
        record_count = len(sub)
        missing_days = total_days - record_count
        missing_pct = (missing_days / total_days) * 100 if total_days > 0 else 0
        outlier_count = sub["is_outlier"].sum() if "is_outlier" in sub.columns else 0
        imputed_count = sub["is_imputed"].sum() if "is_imputed" in sub.columns else 0

        rows.append(
            {
                "Commodity": commodity,
                "DateRange": f"{date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}",
                "TotalDays": total_days,
                "RecordCount": record_count,
                "MissingDays": missing_days,
                "MissingPct": round(missing_pct, 2),
                "OutlierCount": outlier_count,
                "ImputedCount": imputed_count,
                "PriceMean": round(sub[target].mean(), 2),
                "PriceStd": round(sub[target].std(), 2),
            }
        )

    report = pd.DataFrame(rows)

    # Save report
    out_dir = Path(cfg["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "data_quality_report.csv"
    report.to_csv(report_path, index=False)
    logger.info(f"Data quality report saved: {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 9. Vegetable Price Index
# ─────────────────────────────────────────────────────────────────────────────


def create_vegetable_price_index(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    min_commodity_days: int = 365,
) -> pd.DataFrame:
    """
    Create the Kalimati Vegetable Price Index (KVPI).

    Methodology (base-period normalized mean):
        1. Filter to commodities with ≥ ``min_commodity_days`` of data
           (ensures only reliable series contribute to the index).
        2. Calculate a base-period mean for each commodity (first 30 days
           of data), which acts as the commodity's reference price.
        3. Normalise each daily observation: ``norm = (price / base_mean) × 100``.
        4. For each date, compute the cross-sectional mean of normalised
           prices across all contributing commodities → KVPI.

    This is analogous to how stock market indices (e.g., NEPSE) combine
    individual stock prices into a single composite metric.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily data with columns: [Date, Commodity, Average, …].
    cfg : dict
        Pipeline configuration.
    min_commodity_days : int
        Minimum number of data points a commodity must have to be included.

    Returns
    -------
    pd.DataFrame
        Daily index with columns:
        [Date, KVPI, KVPI_Median, N_Commodities, Avg_Price, Min_Price, Max_Price,
         Commodity, Unit, Minimum, Maximum, Average]
        where Commodity='KVPI' for compatibility with downstream pipeline.
    """
    target = cfg["preprocessing"]["target_column"]

    # Step 1: Filter to commodities with sufficient data
    counts = df.groupby("Commodity")["Date"].count()
    valid_commodities = counts[counts >= min_commodity_days].index.tolist()
    df_valid = df[df["Commodity"].isin(valid_commodities)].copy()

    n_excluded = df["Commodity"].nunique() - len(valid_commodities)
    logger.info(
        f"KVPI: Using {len(valid_commodities)} commodities "
        f"(excluded {n_excluded} with <{min_commodity_days} days of data)"
    )

    # Step 2: Base-period mean (first 30 days of EACH commodity's own data)
    # This ensures late-entering commodities use their own initial prices,
    # not the global base period (which they may not appear in).
    first_dates = df_valid.groupby("Commodity")["Date"].min()
    base_records = {}
    for c in valid_commodities:
        c_data = df_valid[df_valid["Commodity"] == c]
        c_base_end = first_dates[c] + pd.Timedelta(days=30)
        c_base_mean = c_data[c_data["Date"] <= c_base_end][target].mean()
        if pd.isna(c_base_mean):
            c_base_mean = c_data[target].mean()
        if pd.isna(c_base_mean) or c_base_mean == 0:
            c_base_mean = 1.0  # Fallback to avoid division by zero
        base_records[c] = c_base_mean
    base_means = pd.Series(base_records)

    # Step 3: Normalise
    df_valid["_base_mean"] = df_valid["Commodity"].map(base_means)
    df_valid["_normalised"] = (df_valid[target] / df_valid["_base_mean"]) * 100

    # Step 4: Daily index
    kvpi = (
        df_valid.groupby("Date")
        .agg(
            KVPI=("_normalised", "mean"),
            KVPI_Median=("_normalised", "median"),
            N_Commodities=("Commodity", "nunique"),
            Avg_Price=(target, "mean"),
            Min_Price=(target, "min"),
            Max_Price=(target, "max"),
        )
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Add compatibility columns for the forecasting pipeline
    kvpi["Commodity"] = "KVPI"
    kvpi["Unit"] = "Index"
    kvpi["Minimum"] = kvpi["Min_Price"]
    kvpi["Maximum"] = kvpi["Max_Price"]
    kvpi["Average"] = kvpi["KVPI"]

    logger.info(
        f"KVPI created: {len(kvpi):,} days, "
        f"{kvpi['Date'].min().date()} → {kvpi['Date'].max().date()}, "
        f"base value ≈ 100, "
        f"current range [{kvpi['KVPI'].min():.1f}, {kvpi['KVPI'].max():.1f}]"
    )

    # Save commodity contribution stats
    out_dir = Path(cfg["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    contrib = pd.DataFrame({
        "Commodity": valid_commodities,
        "Base_Mean_Price": [base_means.get(c, 0) for c in valid_commodities],
        "Total_Days": [counts.get(c, 0) for c in valid_commodities],
    }).sort_values("Commodity")
    contrib.to_csv(out_dir / "kvpi_commodity_contributions.csv", index=False)
    logger.info(f"KVPI commodity contributions saved: {out_dir / 'kvpi_commodity_contributions.csv'}")

    return kvpi


# ─────────────────────────────────────────────────────────────────────────────
# 10. Master Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_preprocessing_pipeline(
    cfg: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Execute the full preprocessing pipeline end-to-end, with caching.

    Uses ALL commodities (no filtering) to build the Kalimati Vegetable
    Price Index (KVPI), a single composite daily time series.

    Data source priority:
    1. Cached CSV (fastest — skip everything)
    2. ``data/processed/full_data.csv`` from ``01_combine_data.py`` (pre-merged)
    3. Individual raw CSVs (original two files)

    Steps (when cache miss or force=True)
    --------------------------------------
    1. Load data (from full_data or raw CSVs)
    2. Normalise schema
    3. Remove duplicates
    4. Detect & treat outliers
    5. Impute missing dates (all commodities)
    6. Create KVPI index
    7. Create aggregations
    8. Save cleaned data
    9. Generate quality report

    Parameters
    ----------
    cfg : dict, optional
        Pipeline config. Loaded from default if None.
    force : bool
        If True, re-run preprocessing even if cache exists.

    Returns
    -------
    tuple
        (kvpi_df, aggregations_dict)
    """
    if cfg is None:
        cfg = load_config()

    ensure_dirs(cfg)

    # ── Cache check ──
    cache_path = Path(cfg["data"]["cleaned_dir"]) / "kalimati_kvpi.csv"
    if cache_path.exists() and not force:
        logger.info(f"✓ Cached KVPI data found: {cache_path}")
        df = pd.read_csv(cache_path)
        df["Date"] = pd.to_datetime(df["Date"])

        logger.info(
            f"  Loaded {len(df):,} records, "
            f"date range {df['Date'].min().date()} → {df['Date'].max().date()}"
        )
        aggregations = {"daily": df}
        return df, aggregations

    # ── Full pipeline ──
    logger.info("=" * 70)
    logger.info("  DATA PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    # Step 1: Load — prefer full_data.csv if it exists
    full_data_path = Path(cfg["data"]["raw_dir"]).parent.parent / "data" / "processed" / "full_data.csv"
    if not full_data_path.exists():
        full_data_path = Path(cfg["data"]["raw_dir"]).parent / "processed" / "full_data.csv"

    if full_data_path.exists():
        logger.info(f"Loading pre-merged data: {full_data_path}")
        raw_df = pd.read_csv(full_data_path, encoding="utf-8")
        logger.info(f"  → {len(raw_df):,} records loaded")
    else:
        logger.info("No pre-merged data found. Loading individual raw CSVs …")
        raw_df = load_raw_csvs(cfg)

    # Drop category columns — not needed for index-based forecasting
    for col in ["Commodity_Category", "Commodity_Group"]:
        if col in raw_df.columns:
            raw_df = raw_df.drop(columns=[col])

    # Step 2: Normalise
    normalised_df = normalise_schema(raw_df)

    # Step 3: Dedup
    deduped_df = remove_duplicates(normalised_df)

    # Step 4: Outliers
    outlier_df = detect_outliers(deduped_df, cfg)

    # Step 5: Impute missing dates for ALL commodities
    imputed_df = impute_missing_dates(outlier_df, cfg)

    n_commodities = imputed_df["Commodity"].nunique()
    logger.info(f"Using ALL {n_commodities} commodities for KVPI construction")

    # Step 6: Create KVPI
    kvpi_df = create_vegetable_price_index(imputed_df, cfg)

    # Step 7: Save
    out_dir = Path(cfg["data"]["cleaned_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save KVPI (primary output)
    kvpi_path = out_dir / "kalimati_kvpi.csv"
    kvpi_df.to_csv(kvpi_path, index=False)
    logger.info(f"Saved KVPI dataset: {kvpi_path}")

    # Also save the full cleaned commodity data for reference
    clean_cols = ["Date", "Commodity", "Unit", "Minimum", "Maximum", "Average"]
    clean_cols = [c for c in clean_cols if c in imputed_df.columns]
    imputed_df[clean_cols].to_csv(out_dir / "kalimati_cleaned_all_commodities.csv", index=False)
    logger.info(f"Saved all-commodity reference: {out_dir / 'kalimati_cleaned_all_commodities.csv'}")

    # Step 8: Quality report
    quality_report = generate_data_quality_report(imputed_df, cfg)
    logger.info(f"\nData Quality Summary (top 10):\n{quality_report.head(10).to_string(index=False)}")

    logger.info("=" * 70)
    logger.info("  PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(
        f"  KVPI: {len(kvpi_df):,} days | "
        f"Built from {kvpi_df['N_Commodities'].median():.0f} commodities/day"
    )

    aggregations = {"daily": kvpi_df}
    return kvpi_df, aggregations


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

    df, aggs = run_preprocessing_pipeline(cfg, force=True)
    print(f"\nKVPI shape: {df.shape}")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"\nFirst 5 rows:\n{df.head()}")

