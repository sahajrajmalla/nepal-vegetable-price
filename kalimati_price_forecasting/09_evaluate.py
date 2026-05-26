#!/usr/bin/env python3
"""
07_evaluate.py — Final Evaluation & Model Comparison Report
=============================================================

Stage 7: Merges all stage results, generates unified comparison table,
bar charts, and identifies the best model.

Usage:
    python 07_evaluate.py

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logger, ensure_dirs
from src.visualization import plot_model_comparison

import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Final Evaluation & Report")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    setup_logger("kalimati.visualization", level=cfg["logging"]["level"])
    ensure_dirs(cfg)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 7: FINAL EVALUATION & COMPARISON REPORT                  ║")
    logger.info("╚" + "═" * 68 + "╝")

    reports_dir = Path(cfg["output"]["reports_dir"])

    # Collect all stage result CSVs
    result_files = {
        "baseline_statistical_results.csv": "Stage 3 (Baselines + Statistical)",
        "ml_results.csv": "Stage 4 (ML)",
        "dl_results.csv": "Stage 5 (DL)",
        "hybrid_results.csv": "Stage 6 (Hybrid)",
        "sota_results.csv": "Stage 8 (SOTA NeuralForecast)",
        "ensemble_results.csv": "Stage 9 (Stacking Ensemble)",
    }

    frames = []
    for fname, desc in result_files.items():
        fpath = reports_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            frames.append(df)
            logger.info(f"  ✓ Loaded: {fname} ({len(df)} rows) — {desc}")
        else:
            logger.warning(f"  ✗ Not found: {fname} — {desc}")

    if not frames:
        logger.error("No result files found! Run stages 3–6 first.")
        return

    all_results = pd.concat(frames, ignore_index=True)

    # Sort
    if "RMSE" in all_results.columns:
        all_results = all_results.sort_values(["Commodity", "Horizon", "RMSE"])

    # Save unified comparison
    unified_path = reports_dir / "model_comparison.csv"
    all_results.to_csv(unified_path, index=False)
    logger.info(f"\n✓ Unified comparison: {unified_path}")

    # Markdown report
    md_path = reports_dir / "model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Kalimati Vegetable Price Index — Model Comparison\n\n")
        f.write("## Methodology\n\n")
        f.write("The **Kalimati Vegetable Price Index (KVPI)** is a composite daily index\n")
        f.write("constructed by normalising each commodity's price relative to its base-period\n")
        f.write("mean (first 30 days) and averaging across all commodities with ≥365 days of data.\n\n")
        f.write("Base value ≈ 100. Higher values indicate price inflation relative to the base period.\n\n")
        f.write("---\n\n")

        for horizon in sorted(all_results["Horizon"].unique()):
            f.write(f"\n## Forecast Horizon: {horizon} days\n\n")
            h_df = all_results[all_results["Horizon"] == horizon].copy()
            display_cols = [c for c in ["Model", "RMSE", "MAE", "MAPE", "sMAPE", "R2"] if c in h_df.columns]
            f.write(h_df[display_cols].to_markdown(index=False))
            f.write("\n")

    logger.info(f"✓ Markdown report: {md_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("  MODEL COMPARISON — KVPI")
    logger.info("=" * 80)
    logger.info(f"\n{all_results.to_string(index=False)}")

    # Comparison charts
    for horizon in all_results["Horizon"].unique():
        try:
            plot_model_comparison(all_results, "KVPI", cfg, metric="RMSE", horizon=horizon)
        except Exception:
            pass
            
    # Diebold-Mariano Tests
    try:
        from src.evaluation import diebold_mariano_test
        from src.utils import sanitize_commodity_name
        slug = sanitize_commodity_name("KVPI")
        
        # We need the true targets
        # Assuming we can find the target from the preprocessing pipeline
        from src.data_preprocessing import run_preprocessing_pipeline
        from src.feature_engineering import engineer_features
        from src.evaluation import fixed_split
        
        kvpi_df, _ = run_preprocessing_pipeline(cfg)
        featured_df = engineer_features(kvpi_df, cfg, commodity="KVPI")
        _, test_df = fixed_split(featured_df, cfg)
        target = cfg["preprocessing"]["target_column"]
        actual = test_df[target].values
        
        # Let's compare Ensemble with best ML/baseline
        ens_pred_file = reports_dir / f"{slug}_stackingensemble_predictions.csv"
        best_base_file = reports_dir / f"{slug}_histgb_predictions.csv" # Defaulting to HistGB as baseline
        
        if ens_pred_file.exists() and best_base_file.exists():
            ens_pred = pd.read_csv(ens_pred_file)["prediction"].values
            base_pred = pd.read_csv(best_base_file)["prediction"].values
            
            n_min = min(len(actual), len(ens_pred), len(base_pred))
            
            logger.info("\n" + "─" * 60)
            logger.info("  DIEBOLD-MARIANO TEST (StackingEnsemble vs HistGB)")
            logger.info("─" * 60)
            
            h = min(cfg["evaluation"]["horizons"])
            res = diebold_mariano_test(actual[:n_min], base_pred[:n_min], ens_pred[:n_min], horizon=h, loss="squared")
            logger.info(f"DM Statistic: {res['dm_stat']:.4f}")
            logger.info(f"p-value     : {res['dm_pvalue']:.4e}")
            if res['dm_pvalue'] < 0.05:
                logger.info("Conclusion  : The difference in accuracy is statistically significant (p < 0.05).")
            else:
                logger.info("Conclusion  : The difference in accuracy is not statistically significant.")
            
            with open(md_path, "a", encoding="utf-8") as f:
                f.write("\n## Statistical Significance (Diebold-Mariano Test)\n\n")
                f.write("Comparing StackingEnsemble vs HistGB:\n")
                f.write(f"- **DM Statistic**: {res['dm_stat']:.4f}\n")
                f.write(f"- **p-value**: {res['dm_pvalue']:.4e}\n")
    except Exception as e:
        logger.warning(f"Failed to run Diebold-Mariano test: {e}")

    # Best model per horizon
    if "RMSE" in all_results.columns:
        logger.info("\n" + "─" * 60)
        logger.info("  BEST MODEL PER FORECAST HORIZON")
        logger.info("─" * 60)
        best = all_results.loc[all_results.groupby("Horizon")["RMSE"].idxmin()]
        for _, row in best.iterrows():
            logger.info(f"  H={int(row['Horizon']):3d}d → {row['Model']:20s} RMSE={row['RMSE']:.4f}")

    logger.info(f"\n✓ Stage 7 complete. All outputs: {reports_dir}/")


if __name__ == "__main__":
    main()
