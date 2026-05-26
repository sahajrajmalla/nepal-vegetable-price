#!/usr/bin/env python3
"""
03_train_baselines.py — Baseline & Statistical Models on KVPI
==============================================================

Stage 3: Runs Naive, Seasonal Naive, Auto-ARIMA, and SARIMA
on the Kalimati Vegetable Price Index.

Usage:
    python 03_train_baselines.py
    python 03_train_baselines.py --skip-eda

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_logger, set_global_seed, ensure_dirs,
    sanitize_commodity_name, timer, log_environment,
)
from src.data_preprocessing import run_preprocessing_pipeline
from src.feature_engineering import engineer_features, get_feature_columns
from src.evaluation import fixed_split, compute_all_metrics
from src.models.baselines import run_baselines
from src.models.statistical import run_statistical_models
from src.visualization import (
    run_eda_plots, plot_forecast_vs_actual, plot_residual_diagnostics,
)

import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Baselines & Statistical")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)

    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    for mod in ["preprocessing", "features", "evaluation", "models.baselines", "models.statistical", "visualization"]:
        setup_logger(f"kalimati.{mod}", level=cfg["logging"]["level"])
    ensure_dirs(cfg)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 3: BASELINE & STATISTICAL MODELS (KVPI)                  ║")
    logger.info("╚" + "═" * 68 + "╝")

    # Load KVPI data
    with timer("Data loading", logger):
        kvpi_df, _ = run_preprocessing_pipeline(cfg)

    commodity = "KVPI"
    featured_df = engineer_features(kvpi_df, cfg, commodity=commodity)

    # EDA
    if not args.skip_eda:
        with timer(f"EDA — {commodity}", logger):
            try:
                run_eda_plots(featured_df, commodity, cfg)
            except Exception as e:
                logger.warning(f"EDA failed: {e}")

    # Split
    train_df, test_df = fixed_split(featured_df, cfg)
    train_y = train_df[cfg["preprocessing"]["target_column"]].dropna().values
    test_y = test_df[cfg["preprocessing"]["target_column"]].dropna().values
    horizons = cfg["evaluation"]["horizons"]
    all_results = []

    # ── Baselines ──
    with timer(f"Baselines — {commodity}", logger):
        try:
            baseline_results = run_baselines(train_y, test_y, cfg)
            for name, res in baseline_results.items():
                pred = res["predictions"]
                for h in horizons:
                    n = min(h, len(test_y), len(pred))
                    if n > 0:
                        m = compute_all_metrics(test_y[:n], pred[:n])
                        all_results.append({"Commodity": commodity, "Model": name, "Horizon": h, **m})
                slug = sanitize_commodity_name(commodity)
                pd.DataFrame({"prediction": pred}).to_csv(
                    Path(cfg["output"]["reports_dir"]) / f"{slug}_{name.lower()}_predictions.csv", index=False
                )
        except Exception as e:
            logger.error(f"Baselines failed: {e}", exc_info=True)

    # ── Statistical ──
    with timer(f"Statistical — {commodity}", logger):
        try:
            stat_results = run_statistical_models(train_y, test_y, cfg)
            for name, res in stat_results.items():
                pred = res["predictions"]
                for h in horizons:
                    n = min(h, len(test_y), len(pred))
                    if n > 0:
                        m = compute_all_metrics(test_y[:n], pred[:n])
                        all_results.append({"Commodity": commodity, "Model": name, "Horizon": h, **m})

                slug = sanitize_commodity_name(commodity)
                pd.DataFrame({"prediction": pred}).to_csv(
                    Path(cfg["output"]["reports_dir"]) / f"{slug}_{name.lower()}_predictions.csv", index=False
                )

                # Plots
                n = min(len(test_y), len(pred))
                test_dates = test_df["Date"].values[:n]
                try:
                    plot_forecast_vs_actual(
                        test_dates, test_y[:n], pred[:n], commodity, name, cfg,
                        lower=res.get("lower", np.zeros(n))[:n],
                        upper=res.get("upper", np.zeros(n))[:n],
                    )
                except Exception:
                    pass
                if res.get("residuals") is not None:
                    try:
                        plot_residual_diagnostics(res["residuals"], name, commodity, cfg)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Statistical failed: {e}", exc_info=True)

    # Save
    if all_results:
        results_df = pd.DataFrame(all_results)
        out_dir = Path(cfg["output"]["reports_dir"])
        results_df.to_csv(out_dir / "baseline_statistical_results.csv", index=False)
        logger.info(f"\n✓ Results saved: baseline_statistical_results.csv")
        logger.info(f"\n{results_df.to_string(index=False)}")

    logger.info("\n✓ Stage 3 complete.")


if __name__ == "__main__":
    main()
