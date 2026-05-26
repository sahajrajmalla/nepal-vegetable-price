#!/usr/bin/env python3
"""
08_train_sota.py — Train NeuralForecast SOTA Models
===================================================

Trains PatchTST and NBEATSx on the KVPI.
Generates multi-step forecasts natively.

Usage:
    python 08_train_sota.py

Author : Sahaj Raj Malla
Created: 2025
"""

import os
import sys
import gc
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logger, set_global_seed, ensure_dirs, sanitize_commodity_name, timer
from src.data_preprocessing import run_preprocessing_pipeline
from src.feature_engineering import engineer_features, get_feature_columns
from src.evaluation import fixed_split, compute_all_metrics

try:
    from src.models.sota_models import train_sota_models, predict_sota
except ImportError:
    pass

import argparse

def main():
    parser = argparse.ArgumentParser(description="Stage 8: SOTA Deep Learning Models")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)

    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    ensure_dirs(cfg)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 8: SOTA DEEP LEARNING (PatchTST, NBEATSx)                ║")
    logger.info("╚" + "═" * 68 + "╝")

    try:
        import neuralforecast
    except ImportError:
        logger.error("neuralforecast is not installed. Please run: pip install neuralforecast")
        sys.exit(1)

    kvpi_df, _ = run_preprocessing_pipeline(cfg)
    commodity = "KVPI"
    featured_df = engineer_features(kvpi_df, cfg, commodity=commodity)
    feature_cols = get_feature_columns(featured_df)

    train_df, test_df = fixed_split(featured_df, cfg)
    
    # We only use a subset of features for neuralforecast exogenous vars
    sota_features = [
        c for c in feature_cols
        if "fest" in c or "weekend" in c or "month" in c or "dayofweek" in c
    ]

    with timer("SOTA models training", logger):
        nf_model = train_sota_models(train_df, cfg, feature_cols=sota_features)
        
    with timer("SOTA models prediction", logger):
        preds_df = predict_sota(nf_model, train_df, test_df, cfg, feature_cols=sota_features)
    
    # preds_df has columns: unique_id, ds, PatchTST, NBEATSx
    # We merge with test_df to calculate metrics
    preds_df["ds"] = pd.to_datetime(preds_df["ds"])
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    
    merged = pd.merge(test_df, preds_df, left_on="Date", right_on="ds", how="inner")
    
    horizons = cfg["evaluation"]["horizons"]
    y_test = merged[cfg["preprocessing"]["target_column"]].values
    
    all_results = []
    reports_dir = Path(cfg["output"]["reports_dir"])
    slug = sanitize_commodity_name(commodity)
    
    for model_name in ["PatchTST", "NBEATSx"]:
        if model_name in merged.columns:
            pred_vals = merged[model_name].values
            
            for h in horizons:
                n = min(h, len(y_test), len(pred_vals))
                if n > 0:
                    m = compute_all_metrics(y_test[:n], pred_vals[:n])
                    all_results.append({"Commodity": commodity, "Model": model_name, "Horizon": h, **m})
            
            # Save predictions
            pd.DataFrame({"prediction": pred_vals}).to_csv(
                reports_dir / f"{slug}_{model_name.lower()}_predictions.csv", index=False
            )
            rmse_val = compute_all_metrics(y_test, pred_vals).get('RMSE')
            if rmse_val is not None:
                logger.info(f"{model_name} — RMSE: {rmse_val:.4f}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(reports_dir / "sota_results.csv", index=False)
        logger.info(f"\n✓ SOTA results saved: sota_results.csv")
        logger.info(f"\n{results_df.to_string(index=False)}")

    logger.info("\n✓ Stage 8 complete.")

if __name__ == "__main__":
    main()
