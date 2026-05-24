#!/usr/bin/env python3
"""
04_train_ml.py — Machine Learning Models on KVPI
=================================================

Trains and evaluates 4 high-performance tree-based regression models on the KVPI:
    1. Random Forest
    2. Extra Trees
    3. HistGradientBoosting (LightGBM equivalent)
    4. XGBoost

Supports recursive out-of-sample multi-step forecasting to prevent lookahead bias.
Optimized for macOS ARM64 stability (n_jobs=1, sequential cleanup, garbage collection).

Usage:
    python 04_train_ml.py

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import gc
import os
import sys
import warnings
from pathlib import Path

# Suppress all library warning outputs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.utils import (
    load_config, setup_logger, set_global_seed, ensure_dirs,
    sanitize_commodity_name, timer,
)
from src.data_preprocessing import run_preprocessing_pipeline
from src.feature_engineering import engineer_features, get_feature_columns
from src.evaluation import fixed_split, compute_all_metrics
from src.models.ml_models import (
    train_random_forest, train_extra_trees, train_hist_gb, train_xgboost,
    get_rf_feature_importance, get_et_feature_importance, get_hg_feature_importance, get_xgb_feature_importance,
    predict_recursive_ml, save_ml_model,
)
from src.visualization import plot_feature_importance, plot_forecast_vs_actual

import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 4: ML Models on KVPI")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)

    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    for mod in ["preprocessing", "features", "evaluation", "models.ml", "visualization"]:
        setup_logger(f"kalimati.{mod}", level=cfg["logging"]["level"])
    ensure_dirs(cfg)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 4: MACHINE LEARNING MODELS (KVPI)                        ║")
    logger.info("╚" + "═" * 68 + "╝")

    with timer("Data loading", logger):
        kvpi_df, _ = run_preprocessing_pipeline(cfg)

    commodity = "KVPI"
    featured_df = engineer_features(kvpi_df, cfg, commodity=commodity)
    feature_cols = get_feature_columns(featured_df)

    train_df, test_df = fixed_split(featured_df, cfg)
    target = cfg["preprocessing"]["target_column"]

    X_train = train_df[feature_cols].dropna(axis=1, how="all")
    y_train = train_df[target]
    X_test = test_df[feature_cols].dropna(axis=1, how="all")
    y_test = test_df[target]

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Drop rows with NaN in training features for clean model fitting
    train_valid = ~X_train.isna().any(axis=1)
    X_train = X_train[train_valid].reset_index(drop=True)
    y_train = y_train[train_valid].reset_index(drop=True)

    feature_names = list(X_train.columns)

    horizons = cfg["evaluation"]["horizons"]
    strategy = cfg["evaluation"].get("strategy", "recursive")
    logger.info(f"Using forecasting strategy: {strategy.upper()}")

    all_results = []
    slug = sanitize_commodity_name(commodity)
    reports_dir = Path(cfg["output"]["reports_dir"])

    ml_cfg = cfg.get("models", {}).get("ml", {})

    # Define model training candidates
    models_to_run = []
    if ml_cfg.get("random_forest", {}).get("enabled", True):
        models_to_run.append(("RandomForest", train_random_forest, get_rf_feature_importance))
    if ml_cfg.get("extra_trees", {}).get("enabled", True):
        models_to_run.append(("ExtraTrees", train_extra_trees, get_et_feature_importance))
    if ml_cfg.get("hist_gb", {}).get("enabled", True):
        models_to_run.append(("HistGB", train_hist_gb, None))
    if ml_cfg.get("xgboost", {}).get("enabled", True):
        models_to_run.append(("XGBoost", train_xgboost, get_xgb_feature_importance))

    with timer(f"ML models training and evaluation", logger):
        for name, train_fn, importance_fn in models_to_run:
            try:
                logger.info(f"\n── Training {name} ──")
                # 1. Fit model
                model, params = train_fn(X_train, y_train, cfg, seed)

                # 2. Forecast
                if strategy == "recursive":
                    logger.info(f"Generating recursive out-of-sample forecast for {name}…")
                    pred = predict_recursive_ml(model, train_df, test_df, feature_names, cfg)
                else:
                    logger.info(f"Generating one-step-ahead rolling forecast for {name}…")
                    pred = model.predict(X_test)

                # 3. Calculate metrics per horizon
                for h in horizons:
                    n = min(h, len(y_test), len(pred))
                    if n > 0:
                        m = compute_all_metrics(y_test.values[:n], pred[:n])
                        all_results.append({"Commodity": commodity, "Model": name, "Horizon": h, **m})

                # Save predictions
                pd.DataFrame({"prediction": pred}).to_csv(
                    reports_dir / f"{slug}_{name.lower()}_predictions.csv", index=False
                )
                rmse_val = compute_all_metrics(y_test.values, pred).get('RMSE')
                if rmse_val is not None:
                    logger.info(f"{name} — RMSE: {rmse_val:.4f}")
                else:
                    logger.info(f"{name} — RMSE: N/A")

                # 4. Feature importance plot
                if importance_fn is not None:
                    importance_df = importance_fn(model, feature_names)
                    try:
                        plot_feature_importance(importance_df, name, commodity, cfg)
                    except Exception:
                        pass
                    del importance_df
                elif name == "HistGB":
                    importance_df = get_hg_feature_importance(model, X_train, y_train, feature_names)
                    try:
                        plot_feature_importance(importance_df, name, commodity, cfg)
                    except Exception:
                        pass
                    del importance_df

                # 5. Forecast plot
                try:
                    n = min(len(y_test), len(pred))
                    plot_forecast_vs_actual(
                        test_df["Date"].values[:n], y_test.values[:n], pred[:n],
                        commodity, name, cfg,
                    )
                except Exception:
                    pass

                # 6. Save model immediately & delete/gc to prevent ARM64 memory segfault
                save_ml_model(model, cfg, commodity, name.lower())
                del model, params
                gc.collect()
                logger.info(f"✓ {name} completed and memory freed.")

            except Exception as e:
                logger.error(f"{name} failed: {e}", exc_info=True)
            gc.collect()

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(reports_dir / "ml_results.csv", index=False)
        logger.info(f"\n✓ ML results saved: ml_results.csv")
        logger.info(f"\n{results_df.to_string(index=False)}")

    logger.info("\n✓ Stage 4 complete.")


if __name__ == "__main__":
    main()
