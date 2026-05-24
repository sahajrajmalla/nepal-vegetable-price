#!/usr/bin/env python3
"""
06_train_hybrid.py — Hybrid Models (ARIMA-LSTM, ARIMA-XGBoost) on KVPI
========================================================================

Stage 6: Combines linear (ARIMA) and nonlinear (LSTM/XGBoost) components.

Usage:
    python 06_train_hybrid.py

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import gc
import os
import sys
import warnings
from pathlib import Path

# ── Suppress ALL warnings & prevent fork segfaults ──
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_logger, set_global_seed, ensure_dirs,
    sanitize_commodity_name, timer,
)
from src.data_preprocessing import run_preprocessing_pipeline
from src.feature_engineering import engineer_features, get_feature_columns
from src.evaluation import fixed_split, compute_all_metrics
from src.models.hybrid import run_hybrid_models

import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Hybrid Models on KVPI")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)

    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    for mod in ["preprocessing", "features", "evaluation", "models.hybrid", "models.statistical", "models.ml", "models.dl"]:
        setup_logger(f"kalimati.{mod}", level=cfg["logging"]["level"])
    ensure_dirs(cfg)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 6: HYBRID MODELS (KVPI)                                  ║")
    logger.info("╚" + "═" * 68 + "╝")

    with timer("Data loading", logger):
        kvpi_df, _ = run_preprocessing_pipeline(cfg)

    commodity = "KVPI"
    featured_df = engineer_features(kvpi_df, cfg, commodity=commodity)
    feature_cols = get_feature_columns(featured_df)

    train_df, test_df = fixed_split(featured_df, cfg)
    target = cfg["preprocessing"]["target_column"]
    train_y = train_df[target].dropna().values
    test_y = test_df[target].dropna().values
    horizons = cfg["evaluation"]["horizons"]
    all_results = []
    slug = sanitize_commodity_name(commodity)
    reports_dir = Path(cfg["output"]["reports_dir"])

    with timer(f"Hybrid models — {commodity}", logger):
        try:
            hybrid_results = run_hybrid_models(
                train_df, test_df, train_y, test_y, cfg, feature_cols, seed
            )
            for name, res in hybrid_results.items():
                pred = res["predictions"]
                yt = res.get("y_test", test_y)

                pd.DataFrame({"prediction": pred}).to_csv(
                    reports_dir / f"{slug}_{name.lower()}_predictions.csv", index=False,
                )

                for h in horizons:
                    n = min(h, len(yt), len(pred))
                    if n > 0:
                        m = compute_all_metrics(yt[:n], pred[:n])
                        all_results.append({"Commodity": commodity, "Model": name, "Horizon": h, **m})

                # Free model memory after saving predictions
                for key in ["arima_model", "lstm_model", "xgb_model"]:
                    if key in res:
                        del res[key]
                gc.collect()

        except Exception as e:
            logger.error(f"Hybrid models failed: {e}", exc_info=True)
        gc.collect()

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(reports_dir / "hybrid_results.csv", index=False)
        logger.info(f"\n✓ Hybrid results saved: hybrid_results.csv")
        logger.info(f"\n{results_df.to_string(index=False)}")

    logger.info("\n✓ Stage 6 complete.")


if __name__ == "__main__":
    main()
