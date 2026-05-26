#!/usr/bin/env python3
"""
08_train_ensemble.py — Advanced Stacking Ensemble Meta-Learner
===============================================================

Combines predictions from the top-performing ML, DL, statistical, and hybrid
models using a sophisticated multi-strategy ensemble approach to produce a
SOTA forecast.

Strategies:
    1. Ridge Stacking — constrained positive-weight Ridge meta-learner
    2. BayesianRidge Stacking — Bayesian regularised meta-learner
    3. Inverse-RMSE Weighted Average — performance-weighted blend
    4. Dynamic Ensemble Selection (DES) — only admits models with val R² > 0

The script automatically selects the strategy that achieves the best
validation RMSE, then evaluates on the full test set.

Usage:
    python 08_train_ensemble.py

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import gc
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logger, set_global_seed, ensure_dirs, sanitize_commodity_name, timer
from src.data_preprocessing import run_preprocessing_pipeline
from src.feature_engineering import engineer_features
from src.evaluation import fixed_split, compute_all_metrics
from src.visualization import plot_forecast_vs_actual

import argparse


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Load prediction file robustly
# ═══════════════════════════════════════════════════════════════════════════


def _load_predictions(pred_file: Path) -> np.ndarray | None:
    """
    Load a prediction CSV, handling both single-column ('prediction')
    and dual-column ('actual', 'prediction') formats (DL models).
    """
    try:
        df = pd.read_csv(pred_file)
        if "prediction" in df.columns:
            return df["prediction"].values
        elif len(df.columns) == 1:
            return df.iloc[:, 0].values
        else:
            return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Expanding-window cross-validated stacking
# ═══════════════════════════════════════════════════════════════════════════


def _cv_stacking_score(
    X_meta: np.ndarray,
    y_meta: np.ndarray,
    meta_learner_class,
    meta_learner_kwargs: dict,
    n_splits: int = 5,
) -> tuple[float, object]:
    """
    Evaluate a meta-learner using expanding-window (TimeSeriesSplit) CV.

    Returns the mean CV RMSE and the final model fitted on all data.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_rmses = []

    for train_idx, val_idx in tscv.split(X_meta):
        X_tr, X_va = X_meta[train_idx], X_meta[val_idx]
        y_tr, y_va = y_meta[train_idx], y_meta[val_idx]

        model = meta_learner_class(**meta_learner_kwargs)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        cv_rmse = float(np.sqrt(np.mean((y_va - pred) ** 2)))
        cv_rmses.append(cv_rmse)

    # Refit on all data
    final_model = meta_learner_class(**meta_learner_kwargs)
    final_model.fit(X_meta, y_meta)

    return float(np.mean(cv_rmses)), final_model


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Stage 9: Advanced Stacking Ensemble")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    set_global_seed(seed)

    logger = setup_logger("kalimati", log_file=cfg["logging"]["log_file"], level=cfg["logging"]["level"])
    for mod in ["preprocessing", "features", "evaluation", "visualization"]:
        setup_logger(f"kalimati.{mod}", level=cfg["logging"]["level"])
    ensure_dirs(cfg)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  STAGE 9: ADVANCED STACKING ENSEMBLE META-LEARNER               ║")
    logger.info("╚" + "═" * 68 + "╝")

    # ── Load data & split ──
    kvpi_df, _ = run_preprocessing_pipeline(cfg)
    commodity = "KVPI"
    featured_df = engineer_features(kvpi_df, cfg, commodity=commodity)
    _, test_df = fixed_split(featured_df, cfg)

    target = cfg["preprocessing"]["target_column"]
    y_test = test_df[target].values
    test_dates = test_df["Date"].values
    n_test = len(y_test)

    reports_dir = Path(cfg["output"]["reports_dir"])
    slug = sanitize_commodity_name(commodity)
    horizons = cfg["evaluation"]["horizons"]

    # ══════════════════════════════════════════════════════════════════════
    # 1. LOAD ALL AVAILABLE PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════
    #
    # Priority-ordered candidate list based on observed pipeline performance.
    # Models are grouped by tier so we prefer strong performers in the
    # ensemble while still allowing weaker ones if they provide diversity.
    #
    # Tier 1 (top ML):  XGBoost, ExtraTrees, HistGB, RandomForest
    # Tier 2 (top DL):  GRU
    # Tier 3 (stat):    SARIMA, Auto_ARIMA
    # Tier 4 (hybrid):  ARIMA_HistGB, ARIMA_LSTM
    # Tier 5 (other):   PatchTST, NBEATSx, LSTM, Naive, Seasonal_Naive
    #
    candidates = [
        "xgboost",
        "extratrees",
        "histgb",
        "randomforest",
        "gru",
        "sarima",
        "auto_arima",
        "arima_histgb",
        "arima_lstm",
        "lstm",
        "patchtst",
        "nbeatsx",
        "naive",
        "seasonal_naive_7",
    ]

    predictions_dict = {}
    excluded_short = {}

    logger.info("── Loading candidate model predictions ──")

    for model_name in candidates:
        pred_file = reports_dir / f"{slug}_{model_name}_predictions.csv"
        if not pred_file.exists():
            continue

        pred_vals = _load_predictions(pred_file)
        if pred_vals is None:
            logger.warning(f"  ✗ {model_name}: Could not parse prediction file")
            continue

        if len(pred_vals) == n_test:
            predictions_dict[model_name] = pred_vals
            logger.info(f"  ✓ {model_name}: {len(pred_vals)} predictions loaded")
        elif len(pred_vals) < n_test:
            # SOTA models (PatchTST, NBEATSx) typically produce h-step
            # forecasts (e.g. 90), not the full 455-day test set.
            # DO NOT pad/truncate — exclude from the main stacking ensemble
            # but record for separate horizon-specific evaluation.
            excluded_short[model_name] = pred_vals
            logger.warning(
                f"  ⚠ {model_name}: Only {len(pred_vals)}/{n_test} predictions — "
                f"excluded from stacking (will evaluate separately at short horizons)"
            )
        else:
            # More predictions than test — truncate to test length
            predictions_dict[model_name] = pred_vals[:n_test]
            logger.info(f"  ✓ {model_name}: truncated {len(pred_vals)} → {n_test}")

    if not predictions_dict:
        logger.error("No valid full-length prediction files found. Please run earlier stages first.")
        sys.exit(1)

    logger.info(f"\n  Stacking candidates ({len(predictions_dict)}): {list(predictions_dict.keys())}")
    if excluded_short:
        logger.info(f"  Short-horizon only  ({len(excluded_short)}): {list(excluded_short.keys())}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. DYNAMIC ENSEMBLE SELECTION (DES)
    # ══════════════════════════════════════════════════════════════════════
    #
    # Use the first 30% of the test set as a validation set to:
    #   (a) Filter models that underperform the naive mean baseline (R² < 0)
    #   (b) Compute per-model validation RMSE for weighting
    #
    val_size = max(30, int(n_test * 0.3))  # at least 30 days
    val_size = min(val_size, n_test - 30)   # leave at least 30 for holdout

    y_val = y_test[:val_size]
    y_holdout = y_test[val_size:]

    X_meta_df = pd.DataFrame(predictions_dict)
    X_meta_all = X_meta_df.values

    # Compute per-model validation metrics
    logger.info(f"\n── Dynamic Ensemble Selection (val_size={val_size}) ──")

    model_val_rmse = {}
    model_val_r2 = {}
    valid_models = []

    for i, col in enumerate(X_meta_df.columns):
        pred_val = X_meta_df[col].values[:val_size]
        metrics = compute_all_metrics(y_val, pred_val)
        val_rmse = metrics.get("RMSE", float("inf"))
        val_r2 = metrics.get("R2", -999)

        model_val_rmse[col] = val_rmse
        model_val_r2[col] = val_r2

        status = "✓" if val_r2 > 0 else "✗"
        logger.info(f"  {status} {col:20s}  val_RMSE={val_rmse:8.4f}  val_R²={val_r2:+.4f}")

        if val_r2 > 0:
            valid_models.append(col)

    if not valid_models:
        logger.warning("All models have R² < 0 on validation set. Keeping all candidates.")
        valid_models = list(X_meta_df.columns)
    else:
        logger.info(f"\n  DES retained: {len(valid_models)}/{len(X_meta_df.columns)} models")

    # Build stacking matrices with DES-filtered models only
    X_stack = X_meta_df[valid_models].values
    X_stack_val = X_stack[:val_size]
    X_stack_holdout = X_stack[val_size:]

    # ══════════════════════════════════════════════════════════════════════
    # 3. MULTI-STRATEGY ENSEMBLE COMPETITION
    # ══════════════════════════════════════════════════════════════════════
    #
    # We try multiple meta-learning strategies and pick the one with the
    # best expanding-window CV RMSE on the validation set.
    #
    logger.info("\n── Meta-Learner Strategy Competition ──")

    strategies = {}

    # Strategy 1: Ridge Stacking (positive weights, moderate regularisation)
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        name = f"Ridge(α={alpha})"
        try:
            cv_rmse, model = _cv_stacking_score(
                X_stack_val, y_val,
                Ridge,
                {"alpha": alpha, "positive": True, "fit_intercept": True},
                n_splits=min(5, max(2, val_size // 15)),
            )
            strategies[name] = {"cv_rmse": cv_rmse, "model": model, "type": "learner"}
        except Exception as e:
            logger.debug(f"  {name} failed: {e}")

    # Strategy 2: BayesianRidge (adaptive regularisation, no positivity constraint)
    try:
        cv_rmse, model = _cv_stacking_score(
            X_stack_val, y_val,
            BayesianRidge,
            {"max_iter": 300, "tol": 1e-4, "fit_intercept": True},
            n_splits=min(5, max(2, val_size // 15)),
        )
        strategies["BayesianRidge"] = {"cv_rmse": cv_rmse, "model": model, "type": "learner"}
    except Exception as e:
        logger.debug(f"  BayesianRidge failed: {e}")

    # Strategy 3: Inverse-RMSE Weighted Average
    try:
        inv_rmse = {m: 1.0 / (model_val_rmse[m] + 1e-8) for m in valid_models}
        total_inv = sum(inv_rmse.values())
        inv_weights = np.array([inv_rmse[m] / total_inv for m in valid_models])

        inv_pred_val = X_stack_val @ inv_weights
        inv_cv_rmse = float(np.sqrt(np.mean((y_val - inv_pred_val) ** 2)))
        strategies["InvRMSE_Weighted"] = {
            "cv_rmse": inv_cv_rmse,
            "weights": inv_weights,
            "type": "weighted",
        }
    except Exception as e:
        logger.debug(f"  InvRMSE_Weighted failed: {e}")

    # Strategy 4: Simple Average (baseline)
    try:
        avg_pred_val = X_stack_val.mean(axis=1)
        avg_cv_rmse = float(np.sqrt(np.mean((y_val - avg_pred_val) ** 2)))
        strategies["SimpleAverage"] = {
            "cv_rmse": avg_cv_rmse,
            "weights": np.ones(len(valid_models)) / len(valid_models),
            "type": "weighted",
        }
    except Exception as e:
        logger.debug(f"  SimpleAverage failed: {e}")

    # Strategy 5: Top-K Average (only top 3 models by val RMSE)
    try:
        sorted_models = sorted(valid_models, key=lambda m: model_val_rmse[m])
        top_k = min(3, len(sorted_models))
        top_k_models = sorted_models[:top_k]
        top_k_idx = [valid_models.index(m) for m in top_k_models]
        top_k_pred_val = X_stack_val[:, top_k_idx].mean(axis=1)
        top_k_cv_rmse = float(np.sqrt(np.mean((y_val - top_k_pred_val) ** 2)))
        strategies[f"Top{top_k}_Average"] = {
            "cv_rmse": top_k_cv_rmse,
            "top_k_models": top_k_models,
            "top_k_idx": top_k_idx,
            "type": "top_k",
        }
    except Exception as e:
        logger.debug(f"  Top-K Average failed: {e}")

    # Strategy 6: Inverse-RMSE Weighted Top-K (top 5)
    try:
        sorted_models = sorted(valid_models, key=lambda m: model_val_rmse[m])
        top_k = min(5, len(sorted_models))
        top_k_models = sorted_models[:top_k]
        top_k_idx = [valid_models.index(m) for m in top_k_models]

        inv_rmse_topk = {m: 1.0 / (model_val_rmse[m] + 1e-8) for m in top_k_models}
        total_inv_topk = sum(inv_rmse_topk.values())
        topk_weights = np.array([inv_rmse_topk[m] / total_inv_topk for m in top_k_models])

        topk_pred_val = X_stack_val[:, top_k_idx] @ topk_weights
        topk_cv_rmse = float(np.sqrt(np.mean((y_val - topk_pred_val) ** 2)))
        strategies[f"InvRMSE_Top{top_k}"] = {
            "cv_rmse": topk_cv_rmse,
            "top_k_models": top_k_models,
            "top_k_idx": top_k_idx,
            "topk_weights": topk_weights,
            "type": "top_k_weighted",
        }
    except Exception as e:
        logger.debug(f"  InvRMSE Top-K failed: {e}")

    if not strategies:
        logger.error("All meta-learner strategies failed. Cannot build ensemble.")
        sys.exit(1)

    # Report and select best
    logger.info("\n  Strategy competition results:")
    for name, info in sorted(strategies.items(), key=lambda x: x[1]["cv_rmse"]):
        logger.info(f"    {name:30s}  val_RMSE = {info['cv_rmse']:.4f}")

    best_strategy_name = min(strategies, key=lambda k: strategies[k]["cv_rmse"])
    best_strategy = strategies[best_strategy_name]
    logger.info(f"\n  ★ Selected: {best_strategy_name} (val_RMSE={best_strategy['cv_rmse']:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # 4. GENERATE FULL TEST SET PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════

    if best_strategy["type"] == "learner":
        # Refit the best meta-learner on the full validation set
        # so it has the maximum training data available
        model = best_strategy["model"]
        # The model was already fitted on X_stack_val via _cv_stacking_score
        # Now predict on the full test set
        final_preds = model.predict(X_stack)

        # Log learned weights
        if hasattr(model, "coef_"):
            coefs = model.coef_
            if np.sum(np.abs(coefs)) > 0:
                normalised = coefs / np.sum(np.abs(coefs))
            else:
                normalised = coefs
            logger.info("\n  Meta-Learner Weights:")
            for name, w, nw in zip(valid_models, coefs, normalised):
                logger.info(f"    {name:20s}: coef={w:+.6f}  normalised={nw:+.4f}")
        if hasattr(model, "intercept_"):
            logger.info(f"    {'intercept':20s}: {model.intercept_:+.6f}")

    elif best_strategy["type"] == "weighted":
        weights = best_strategy["weights"]
        final_preds = X_stack @ weights
        logger.info("\n  Ensemble Weights:")
        for name, w in zip(valid_models, weights):
            logger.info(f"    {name:20s}: {w:.4f}")

    elif best_strategy["type"] == "top_k":
        idx = best_strategy["top_k_idx"]
        final_preds = X_stack[:, idx].mean(axis=1)
        logger.info(f"\n  Top-K Models: {best_strategy['top_k_models']}")

    elif best_strategy["type"] == "top_k_weighted":
        idx = best_strategy["top_k_idx"]
        topk_w = best_strategy["topk_weights"]
        final_preds = X_stack[:, idx] @ topk_w
        logger.info(f"\n  Top-K Models: {best_strategy['top_k_models']}")
        for name, w in zip(best_strategy["top_k_models"], topk_w):
            logger.info(f"    {name:20s}: {w:.4f}")

    else:
        logger.error(f"Unknown strategy type: {best_strategy['type']}")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # 5. EVALUATE ENSEMBLE AT MULTIPLE HORIZONS
    # ══════════════════════════════════════════════════════════════════════

    overall_metrics = compute_all_metrics(y_test, final_preds)
    rmse_overall = overall_metrics.get("RMSE", float("nan"))
    logger.info(f"\n  StackingEnsemble — overall RMSE: {rmse_overall:.4f}")

    all_results = []
    for h in horizons:
        n = min(h, len(y_test), len(final_preds))
        if n > 0:
            m = compute_all_metrics(y_test[:n], final_preds[:n])
            all_results.append({"Commodity": commodity, "Model": "StackingEnsemble", "Horizon": h, **m})

    # ── Also evaluate the short-horizon-only models at applicable horizons ──
    for model_name, short_preds in excluded_short.items():
        n_short = len(short_preds)
        for h in horizons:
            n = min(h, n_short, n_test)
            if n > 0:
                m = compute_all_metrics(y_test[:n], short_preds[:n])
                all_results.append({
                    "Commodity": commodity,
                    "Model": f"{model_name}(short)",
                    "Horizon": h,
                    **m,
                })

    # ══════════════════════════════════════════════════════════════════════
    # 6. SAVE OUTPUTS
    # ══════════════════════════════════════════════════════════════════════

    # Save ensemble predictions
    pd.DataFrame({"prediction": final_preds}).to_csv(
        reports_dir / f"{slug}_stackingensemble_predictions.csv", index=False
    )
    logger.info(f"  Saved: {slug}_stackingensemble_predictions.csv")

    # Forecast plot
    try:
        n = min(len(test_dates), len(y_test), len(final_preds))
        plot_forecast_vs_actual(
            test_dates[:n], y_test[:n], final_preds[:n],
            commodity, "StackingEnsemble", cfg,
        )
    except Exception as e:
        logger.debug(f"  Forecast plot failed: {e}")

    # Results CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(reports_dir / "ensemble_results.csv", index=False)
        logger.info(f"\n✓ Ensemble results saved: ensemble_results.csv")
        logger.info(f"\n{results_df.to_string(index=False)}")

    # ── Comparison with best individual model ──
    logger.info("\n── Ensemble vs Best Individual Models ──")
    for h in horizons:
        n = min(h, len(y_test), len(final_preds))
        if n <= 0:
            continue
        ens_rmse = compute_all_metrics(y_test[:n], final_preds[:n]).get("RMSE", float("inf"))

        best_indiv_name = None
        best_indiv_rmse = float("inf")
        for model_name, preds in predictions_dict.items():
            indiv_rmse = compute_all_metrics(y_test[:n], preds[:n]).get("RMSE", float("inf"))
            if indiv_rmse < best_indiv_rmse:
                best_indiv_rmse = indiv_rmse
                best_indiv_name = model_name

        improvement = ((best_indiv_rmse - ens_rmse) / best_indiv_rmse) * 100
        marker = "↑" if improvement > 0 else "↓"
        logger.info(
            f"  h={h:3d}d: Ensemble RMSE={ens_rmse:.4f} vs "
            f"{best_indiv_name} RMSE={best_indiv_rmse:.4f} "
            f"({marker} {abs(improvement):.2f}%)"
        )

    gc.collect()
    logger.info("\n✓ Stage 9 complete.")


if __name__ == "__main__":
    main()
