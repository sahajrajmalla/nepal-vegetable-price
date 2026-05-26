#!/usr/bin/env python3
"""
02_preprocess.py — Data Preprocessing & KVPI Index Creation
=============================================================

Stage 2: Cleans ALL commodity data and creates the Kalimati Vegetable
Price Index (KVPI) — a single daily composite index from 100+ commodities.

Usage:
    python 02_preprocess.py                  # Use cache if available
    python 02_preprocess.py --force           # Force re-processing

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logger, set_global_seed, ensure_dirs, timer, log_environment
from src.data_preprocessing import run_preprocessing_pipeline

import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Preprocess & Create KVPI")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["random_seed"])

    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    setup_logger("kalimati.preprocessing", level=cfg["logging"]["level"])
    ensure_dirs(cfg)
    log_environment(logger)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 2: DATA PREPROCESSING & KVPI INDEX                        ║")
    logger.info("╚" + "═" * 68 + "╝")

    with timer("Data Preprocessing & KVPI", logger):
        kvpi_df, _ = run_preprocessing_pipeline(cfg, force=args.force)

    logger.info(f"\n✓ KVPI: {len(kvpi_df):,} days")
    logger.info(f"  Date range: {kvpi_df['Date'].min().date()} → {kvpi_df['Date'].max().date()}")
    logger.info(f"  Output: outputs/cleaned_data/kalimati_kvpi.csv")


if __name__ == "__main__":
    main()
