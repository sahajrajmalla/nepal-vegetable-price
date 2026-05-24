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
