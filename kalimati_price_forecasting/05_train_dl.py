#!/usr/bin/env python3
"""
05_train_dl.py — Deep Learning Models (LSTM, GRU) on KVPI — PyTorch
=====================================================================

Stage 5: Trains LSTM and GRU on the KVPI index. Saves training
history CSVs, loss curve plots, and model weights.

Usage:
    python 05_train_dl.py

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

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
from src.evaluation import compute_all_metrics
from src.models.dl_models import run_dl_models, save_dl_model, save_training_history
from src.visualization import plot_training_history, plot_forecast_vs_actual

import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 5: DL Models on KVPI")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)

    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    for mod in ["preprocessing", "features", "evaluation", "models.dl", "visualization"]:
        setup_logger(f"kalimati.{mod}", level=cfg["logging"]["level"])
    ensure_dirs(cfg)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 5: DEEP LEARNING MODELS — PyTorch (KVPI)                 ║")
    logger.info("╚" + "═" * 68 + "╝")

    with timer("Data loading", logger):
        kvpi_df, _ = run_preprocessing_pipeline(cfg)

    commodity = "KVPI"
    featured_df = engineer_features(kvpi_df, cfg, commodity=commodity)
    feature_cols = get_feature_columns(featured_df)
    horizons = cfg["evaluation"]["horizons"]
    all_results = []

    with timer(f"DL models — {commodity}", logger):
        try:
            dl_results = run_dl_models(featured_df, cfg, feature_cols, seed)
            for name, res in dl_results.items():
                pred = res["predictions"]
                yt = res["y_test"]

                slug = sanitize_commodity_name(commodity)
                pd.DataFrame({"actual": yt, "prediction": pred}).to_csv(
                    Path(cfg["output"]["reports_dir"]) / f"{slug}_{name.lower()}_predictions.csv", index=False,
                )

                for h in horizons:
                    n = min(h, len(yt), len(pred))
                    if n > 0:
                        m = compute_all_metrics(yt[:n], pred[:n])
                        all_results.append({"Commodity": commodity, "Model": name, "Horizon": h, **m})

                if "history" in res and res["history"]:
                    try:
                        save_training_history(res["history"], cfg, commodity, name.lower())
                    except Exception:
                        pass
                    try:
                        plot_training_history(res["history"], name, commodity, cfg)
                    except Exception:
                        pass
                try:
                    save_dl_model(res["model"], cfg, commodity, name.lower())
                except Exception:
                    pass
                try:
                    n = min(len(yt), len(pred))
                    plot_forecast_vs_actual(np.arange(n), yt[:n], pred[:n], commodity, name, cfg)
                except Exception:
                    pass

                rmse_val = res['metrics'].get('RMSE')
                if rmse_val is not None:
                    logger.info(f"  {name} — RMSE: {rmse_val:.4f}")
                else:
                    logger.info(f"  {name} — RMSE: N/A")
        except Exception as e:
            logger.error(f"DL models failed: {e}", exc_info=True)

    if all_results:
        results_df = pd.DataFrame(all_results)
        out_dir = Path(cfg["output"]["reports_dir"])
        results_df.to_csv(out_dir / "dl_results.csv", index=False)
        logger.info(f"\n✓ DL results saved: dl_results.csv")
        logger.info(f"\n{results_df.to_string(index=False)}")

    logger.info("\n✓ Stage 5 complete.")


if __name__ == "__main__":
    main()
