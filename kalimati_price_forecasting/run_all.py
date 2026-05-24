#!/usr/bin/env python3
"""
run_all.py — Master Pipeline for Kalimati Vegetable Price Forecasting
======================================================================

Orchestrates the full forecasting pipeline end-to-end:

    1. Data preprocessing (with caching — skips if cleaned data exists)
    2. Feature engineering (lags, rolling, calendar, festivals)
    3. EDA visualizations (publication-quality, 300 DPI)
    4. Model training with Optuna hyperparameter optimisation
    5. Evaluation (multi-horizon metrics, residual diagnostics)
    6. Explainability (SHAP, feature importance)
    7. Summary report generation

Usage
-----
    python run_all.py                                       # All commodities
    python run_all.py --commodity "Tomato Big(Nepali)"       # Single commodity
    python run_all.py --skip-dl                              # Skip deep learning
    python run_all.py --force-preprocess                     # Re-run preprocessing

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config,
    setup_logger,
    set_global_seed,
    ensure_dirs,
    get_selected_commodities,
    sanitize_commodity_name,
    timer,
    log_environment,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe metric computation at multiple horizons
# ─────────────────────────────────────────────────────────────────────────────


def _compute_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list,
    commodity: str,
    model_name: str,
    compute_fn,
) -> list:
    """
    Safely compute metrics at multiple horizons, handling length mismatches.

    Always uses ``min(h, len(y_true), len(y_pred))`` to avoid shape errors.
    """
    results = []
    for h in horizons:
        n = min(h, len(y_true), len(y_pred))
        if n == 0:
            continue
        m = compute_fn(y_true[:n], y_pred[:n])
        if m:
            results.append({
                "Commodity": commodity, "Model": model_name, "Horizon": h, **m
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalimati Vegetable Price Forecasting Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--commodity", type=str, default=None,
        help='Single commodity to forecast (e.g., "Tomato Big(Nepali)")'
    )
    parser.add_argument(
        "--skip-dl", action="store_true",
        help="Skip deep learning models (faster execution)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA visualization generation"
    )
    parser.add_argument(
        "--force-preprocess", action="store_true",
        help="Force re-run preprocessing (ignore cache)"
    )
    parser.add_argument(
        "--horizons", type=str, default=None,
        help="Comma-separated horizons (e.g., 7,14,30)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Per-commodity pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline_for_commodity(
    commodity: str,
    df: pd.DataFrame,
    cfg: dict,
    logger,
    skip_dl: bool = False,
    skip_eda: bool = False,
    horizons: list = None,
) -> list:
    """Run the full modelling pipeline for a single commodity."""
    from src.feature_engineering import engineer_features, get_feature_columns
    from src.evaluation import (
        fixed_split, compute_all_metrics, evaluate_horizons,
        residual_diagnostics, bootstrap_prediction_intervals,
    )
    from src.models.baselines import run_baselines
    from src.models.statistical import run_statistical_models
    from src.models.ml_models import run_ml_models, save_ml_model
    from src.models.dl_models import run_dl_models, save_dl_model
    from src.models.hybrid import run_hybrid_models
    from src.visualization import (
        run_eda_plots, plot_forecast_vs_actual, plot_all_models_forecast,
        plot_residual_diagnostics, plot_feature_importance,
        plot_shap_summary, plot_training_history, plot_model_comparison,
    )

    target = cfg["preprocessing"]["target_column"]
    if horizons is None:
        horizons = cfg["evaluation"]["horizons"]
    seed = cfg["project"]["random_seed"]
    all_results = []

    logger.info("=" * 70)
    logger.info(f"  COMMODITY: {commodity}")
    logger.info("=" * 70)

    # ── 1. Filter & Feature Engineering ──
    sub = df[df["Commodity"] == commodity].copy()
    if sub.empty:
        logger.error(f"No data for '{commodity}'. Skipping.")
        return all_results

    with timer(f"Feature engineering — {commodity}", logger):
        featured_df = engineer_features(sub, cfg, commodity=commodity)

    # ── 2. EDA Plots ──
    if not skip_eda:
        with timer(f"EDA plots — {commodity}", logger):
            try:
                run_eda_plots(featured_df, commodity, cfg)
            except Exception as e:
                logger.warning(f"EDA plots failed: {e}")

    # ── 3. Train/Test Split ──
    train_df, test_df = fixed_split(featured_df, cfg)
    if test_df.empty or len(train_df) < 100:
        logger.error(f"Insufficient data after split for {commodity}. Skipping.")
        return all_results

    # ── Prepare datasets ──
    feature_cols = get_feature_columns(featured_df)
    train_clean = train_df[["Date", target] + feature_cols].dropna()
    test_clean = test_df[["Date", target] + feature_cols].dropna()

    if train_clean.empty or test_clean.empty:
        logger.error(f"Empty dataset after NaN removal for {commodity}. Skipping.")
        return all_results

    X_train = train_clean[feature_cols]
    y_train = train_clean[target]
    X_test = test_clean[feature_cols]
    y_test = test_clean[target]
    test_dates = test_clean["Date"].values
    train_y_raw = train_df[target].dropna().values
    test_y_raw = test_df[target].dropna().values

    model_predictions = {}  # name → predictions array (for composite plot)

    # Progress bar for model stages
    stages = ["Baselines", "Statistical", "ML Models"]
    if not skip_dl:
        stages.append("DL Models")
    stages.append("Hybrid Models")
    pbar = tqdm(stages, desc=f"  {commodity}", unit="stage", leave=True)

    # ─────────────────────────────────────────────────────────────────────
    # 4. BASELINES
    # ─────────────────────────────────────────────────────────────────────
    pbar.set_description(f"  {commodity} → Baselines")
    with timer(f"Baselines — {commodity}", logger):
        try:
            baseline_results = run_baselines(train_y_raw, test_y_raw, cfg)
            for name, res in baseline_results.items():
                pred = res["predictions"]
                model_predictions[name] = pred
                all_results.extend(_compute_horizon_metrics(
                    test_y_raw, pred, horizons, commodity, name,
                    compute_all_metrics,
                ))
        except Exception as e:
            logger.error(f"Baselines failed: {e}", exc_info=True)
    pbar.update(1)

    # ─────────────────────────────────────────────────────────────────────
    # 5. STATISTICAL (ARIMA / SARIMA)
    # ─────────────────────────────────────────────────────────────────────
    pbar.set_description(f"  {commodity} → Statistical")
    with timer(f"Statistical models — {commodity}", logger):
        try:
            stat_results = run_statistical_models(train_y_raw, test_y_raw, cfg)
            for name, res in stat_results.items():
                pred = res["predictions"]
                model_predictions[name] = pred
                all_results.extend(_compute_horizon_metrics(
                    test_y_raw, pred, horizons, commodity, name,
                    compute_all_metrics,
                ))
                # Forecast plot with CI
                if res.get("lower") is not None:
                    n = min(len(test_dates), len(pred), len(test_y_raw))
                    try:
                        plot_forecast_vs_actual(
                            test_dates[:n], test_y_raw[:n], pred[:n],
                            commodity, name, cfg,
                            lower=res["lower"][:n], upper=res["upper"][:n],
                        )
                    except Exception as e:
                        logger.debug(f"Forecast plot failed for {name}: {e}")
                # Residual diagnostics
                if res.get("residuals") is not None:
                    try:
                        plot_residual_diagnostics(
                            res["residuals"], name, commodity, cfg
                        )
                    except Exception as e:
                        logger.debug(f"Residual plot failed for {name}: {e}")
        except Exception as e:
            logger.error(f"Statistical models failed: {e}", exc_info=True)
    pbar.update(1)

    # ─────────────────────────────────────────────────────────────────────
    # 6. ML MODELS (Random Forest, XGBoost — Optuna HPO)
    # ─────────────────────────────────────────────────────────────────────
    pbar.set_description(f"  {commodity} → ML (Optuna)")
    with timer(f"ML models — {commodity}", logger):
        try:
            ml_results = run_ml_models(
                train_df, test_df, X_train, y_train, X_test, y_test, cfg, feature_cols, seed
            )
            for name, res in ml_results.items():
                pred = res["predictions"]
                model_predictions[name] = pred
                all_results.extend(_compute_horizon_metrics(
                    y_test.values, pred, horizons, commodity, name,
                    compute_all_metrics,
                ))
                # Save model
                try:
                    save_ml_model(res["model"], cfg, commodity, name.lower())
                except Exception:
                    pass
                # Feature importance
                if "feature_importance" in res:
                    try:
                        plot_feature_importance(
                            res["feature_importance"], name, commodity, cfg
                        )
                    except Exception:
                        pass
                # NOTE: SHAP disabled — TreeExplainer causes segfault on macOS ARM.
                # Feature importance plots above provide equivalent explainability.
        except Exception as e:
            logger.error(f"ML models failed: {e}", exc_info=True)
    pbar.update(1)

    # ─────────────────────────────────────────────────────────────────────
    # 7. DEEP LEARNING (LSTM, GRU)
    # ─────────────────────────────────────────────────────────────────────
    if not skip_dl:
        pbar.set_description(f"  {commodity} → Deep Learning")
        with timer(f"DL models — {commodity}", logger):
            try:
                dl_results = run_dl_models(featured_df, cfg, feature_cols, seed)
                for name, res in dl_results.items():
                    pred = res["predictions"]
                    yt = res["y_test"]
                    model_predictions[name] = pred
                    all_results.extend(_compute_horizon_metrics(
                        yt, pred, horizons, commodity, name,
                        compute_all_metrics,
                    ))
                    # Save training history as CSV
                    if "history" in res and res["history"] is not None:
                        try:
                            from src.models.dl_models import save_training_history
                            save_training_history(
                                res["history"], cfg, commodity, name.lower()
                            )
                        except Exception:
                            pass
                        try:
                            plot_training_history(
                                res["history"], name, commodity, cfg
                            )
                        except Exception:
                            pass
                    try:
                        save_dl_model(
                            res["model"], cfg, commodity, name.lower()
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"DL models failed: {e}", exc_info=True)
        pbar.update(1)

    # ─────────────────────────────────────────────────────────────────────
    # 8. HYBRID MODELS (ARIMA-LSTM, ARIMA-XGBoost)
    # ─────────────────────────────────────────────────────────────────────
    pbar.set_description(f"  {commodity} → Hybrid")
    with timer(f"Hybrid models — {commodity}", logger):
        try:
            hybrid_results = run_hybrid_models(
                train_clean, test_clean, train_y_raw, test_y_raw,
                cfg, feature_cols, seed
            )
            for name, res in hybrid_results.items():
                pred = res["predictions"]
                yt = res.get("y_test", test_y_raw)
                model_predictions[name] = pred
                all_results.extend(_compute_horizon_metrics(
                    yt, pred, horizons, commodity, name,
                    compute_all_metrics,
                ))
        except Exception as e:
            logger.error(f"Hybrid models failed: {e}", exc_info=True)
    pbar.update(1)
    pbar.close()

    # ─────────────────────────────────────────────────────────────────────
    # 9. ALL-MODELS FORECAST PLOT
    # ─────────────────────────────────────────────────────────────────────
    if model_predictions:
        try:
            n = min(
                len(test_dates),
                len(test_y_raw),
                min(len(p) for p in model_predictions.values()),
            )
            if n > 0:
                trimmed = {k: v[:n] for k, v in model_predictions.items()}
                plot_all_models_forecast(
                    test_dates[:n], test_y_raw[:n], trimmed, commodity, cfg
                )
        except Exception as e:
            logger.debug(f"All-models plot failed: {e}")

    logger.info(
        f"✓ Completed {commodity}: "
        f"{len(model_predictions)} models, "
        f"{len(all_results)} result entries"
    )
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)

    # Setup logging — only the root logger gets handlers
    logger = setup_logger(
        "kalimati",
        log_file=cfg["logging"]["log_file"],
        level=cfg["logging"]["level"],
    )
    # Child loggers: just register their level, no handlers
    for module in [
        "preprocessing", "features", "evaluation",
        "models.baselines", "models.statistical",
        "models.ml", "models.dl", "models.hybrid",
        "visualization",
    ]:
        setup_logger(
            f"kalimati.{module}",
            level=cfg["logging"]["level"],
        )

    ensure_dirs(cfg)
    log_environment(logger)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  KALIMATI VEGETABLE PRICE FORECASTING PIPELINE                    ║")
    logger.info("╚" + "═" * 68 + "╝")

    # ── Step 1: Preprocessing (with caching) ──
    from src.data_preprocessing import run_preprocessing_pipeline
    with timer("Data Preprocessing", logger):
        cleaned_df, aggregations = run_preprocessing_pipeline(
            cfg, force=args.force_preprocess
        )

    # ── Step 2: Determine commodities ──
    if args.commodity:
        commodities = [args.commodity]
    else:
        commodities = get_selected_commodities(cfg)

    available = cleaned_df["Commodity"].unique()
    commodities = [c for c in commodities if c in available]

    if not commodities:
        logger.error("No valid commodities found. Check config or --commodity flag.")
        sys.exit(1)

    logger.info(f"Processing {len(commodities)} commodities: {commodities}")

    # Parse horizons
    horizons = None
    if args.horizons:
        horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # ── Step 3: Run pipeline per commodity ──
    all_results = []
    for commodity in tqdm(commodities, desc="Pipeline", unit="commodity"):
        try:
            results = run_pipeline_for_commodity(
                commodity, cleaned_df, cfg, logger,
                skip_dl=args.skip_dl,
                skip_eda=args.skip_eda,
                horizons=horizons,
            )
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Pipeline failed for {commodity}: {e}", exc_info=True)

    # ── Step 4: Summary Report ──
    if all_results:
        from src.evaluation import generate_results_table, rank_models, save_results
        from src.visualization import plot_model_comparison

        results_df = generate_results_table(all_results, cfg)
        ranked_df = rank_models(results_df, metric="RMSE")
        save_results(ranked_df, cfg, filename="model_comparison")

        # Comparison plots per commodity
        for commodity in commodities:
            for h in (horizons or cfg["evaluation"]["horizons"]):
                try:
                    plot_model_comparison(
                        ranked_df, commodity, cfg, metric="RMSE", horizon=h
                    )
                except Exception:
                    pass

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("  FINAL RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\n{ranked_df.to_string(index=False)}")

        # Best models
        logger.info("\n── Best Models (by RMSE) ──")
        for commodity in commodities:
            for h in (horizons or cfg["evaluation"]["horizons"]):
                sub = ranked_df[
                    (ranked_df["Commodity"] == commodity) &
                    (ranked_df["Horizon"] == h)
                ]
                if not sub.empty:
                    best = sub.iloc[0]
                    logger.info(
                        f"  {commodity:30s} h={h:3d}: "
                        f"{best['Model']:20s} RMSE={best['RMSE']:.4f}"
                    )
    else:
        logger.warning("No results generated. Check data and configuration.")

    logger.info("\n╔" + "═" * 68 + "╗")
    logger.info("║  PIPELINE COMPLETE                                               ║")
    logger.info("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
