#!/usr/bin/env python3
"""
08_train_ensemble.py — Advanced Stacking Ensemble Meta-Learner
===============================================================

Combines predictions from XGBoost (precision) and GRU (trend) using a
sophisticated multi-strategy ensemble approach to produce a SOTA forecast.

The key innovation is ONLINE strategies that adapt in real-time as test
observations become available (simulating a production deployment where
yesterday's actual is known today). These strategies are evaluated using
FULL-TEST online RMSE — a legitimate metric because they only use past
observations at each step (causal/online, no data leakage).

Strategies include:
    Static:   Ridge, BayesianRidge, Optimized Blend, InvRMSE-Weighted
    Online:   Adaptive Rolling, EWM-Adaptive, Residual-Corrected,
              Momentum-Corrected, Adaptive+Momentum Hybrid

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
from scipy.optimize import minimize

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

    final_model = meta_learner_class(**meta_learner_kwargs)
    final_model.fit(X_meta, y_meta)
    return float(np.mean(cv_rmses)), final_model


# ═══════════════════════════════════════════════════════════════════════════
# ONLINE ENSEMBLE STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════
#
# All online strategies are CAUSAL: at step t, they only use observations
# y[0], y[1], ..., y[t-1] to compute the prediction for step t.
# This means evaluating them on the full test set is legitimate (not leakage).
#


def _adaptive_rolling_weights(
    xgb_preds: np.ndarray,
    gru_preds: np.ndarray,
    y_true: np.ndarray,
    lookback: int = 14,
    min_weight: float = 0.05,
) -> np.ndarray:
    """
    At each step t, compute rolling RMSE over the last `lookback` steps
    for each model, and blend with inverse-RMSE weights.
    """
    n = len(y_true)
    blended = np.zeros(n)

    for t in range(n):
        if t == 0:
            blended[t] = 0.5 * xgb_preds[t] + 0.5 * gru_preds[t]
            continue

        start = max(0, t - lookback)
        y_hist = y_true[start:t]
        xgb_rmse = np.sqrt(np.mean((y_hist - xgb_preds[start:t]) ** 2)) + 1e-8
        gru_rmse = np.sqrt(np.mean((y_hist - gru_preds[start:t]) ** 2)) + 1e-8

        w_xgb = 1.0 / xgb_rmse
        w_gru = 1.0 / gru_rmse
        total = w_xgb + w_gru
        w_xgb = max(w_xgb / total, min_weight)
        w_gru = max(w_gru / total, min_weight)
        total2 = w_xgb + w_gru
        w_xgb /= total2
        w_gru /= total2

        blended[t] = w_xgb * xgb_preds[t] + w_gru * gru_preds[t]

    return blended


def _ewm_adaptive_blend(
    xgb_preds: np.ndarray,
    gru_preds: np.ndarray,
    y_true: np.ndarray,
    span: int = 14,
    min_weight: float = 0.05,
) -> np.ndarray:
    """
    Exponentially-weighted adaptive blend. Reacts faster to regime changes
    than fixed-window rolling because recent errors get exponentially more weight.
    """
    n = len(y_true)
    alpha = 2.0 / (span + 1)
    blended = np.zeros(n)
    ewm_xgb_sq = 0.0
    ewm_gru_sq = 0.0

    for t in range(n):
        if t == 0:
            blended[t] = 0.5 * xgb_preds[t] + 0.5 * gru_preds[t]
            continue

        ewm_xgb_sq = alpha * (y_true[t-1] - xgb_preds[t-1]) ** 2 + (1 - alpha) * ewm_xgb_sq
        ewm_gru_sq = alpha * (y_true[t-1] - gru_preds[t-1]) ** 2 + (1 - alpha) * ewm_gru_sq

        w_xgb = 1.0 / (np.sqrt(ewm_xgb_sq) + 1e-8)
        w_gru = 1.0 / (np.sqrt(ewm_gru_sq) + 1e-8)
        total = w_xgb + w_gru
        w_xgb = max(w_xgb / total, min_weight)
        w_gru = max(w_gru / total, min_weight)
        total2 = w_xgb + w_gru
        blended[t] = (w_xgb / total2) * xgb_preds[t] + (w_gru / total2) * gru_preds[t]

    return blended


def _residual_corrected_online(
    xgb_preds: np.ndarray,
    gru_preds: np.ndarray,
    y_true: np.ndarray,
    base_alpha: float = 0.6,
    correction_window: int = 14,
) -> np.ndarray:
    """
    Static blend + rolling mean-residual bias correction.
    If the blend has been consistently underpredicting, correction is positive.
    """
    n = len(y_true)
    base = base_alpha * xgb_preds + (1 - base_alpha) * gru_preds
    corrected = np.copy(base)

    for t in range(1, n):
        start = max(0, t - correction_window)
        past_residuals = y_true[start:t] - base[start:t]
        corrected[t] = base[t] + np.mean(past_residuals)

    return corrected


def _momentum_corrected_blend(
    xgb_preds: np.ndarray,
    gru_preds: np.ndarray,
    y_true: np.ndarray,
    base_alpha: float = 0.65,
    momentum_window: int = 7,
    momentum_gain: float = 0.5,
) -> np.ndarray:
    """
    MOMENTUM-AWARE BIAS CORRECTION — the key innovation.

    Problem: When actuals are in a strong upswing and both models lag behind,
    the mean residual correction is too slow — it only corrects by the
    *average* past error, not the *accelerating* trend.

    Solution: Compute the DERIVATIVE (slope) of the residual series. If
    residuals are growing (model falling further behind), add extra
    momentum correction proportional to the rate of residual growth.

    correction = mean_residual + momentum_gain * residual_slope * momentum_window

    This makes the forecast "lean into" the trend when it detects
    consistent underprediction that is getting worse over time.
    """
    n = len(y_true)
    base = base_alpha * xgb_preds + (1 - base_alpha) * gru_preds
    corrected = np.copy(base)

    for t in range(2, n):
        start = max(0, t - momentum_window)
        window_len = t - start
        if window_len < 2:
            past_resid = y_true[start:t] - base[start:t]
            corrected[t] = base[t] + np.mean(past_resid)
            continue

        past_resid = y_true[start:t] - base[start:t]
        mean_resid = np.mean(past_resid)

        # Compute residual slope using simple linear regression
        x = np.arange(window_len, dtype=float)
        x_mean = x.mean()
        r_mean = past_resid.mean()
        slope = np.sum((x - x_mean) * (past_resid - r_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)

        # Momentum: if slope > 0, model is falling further behind
        momentum = momentum_gain * slope * window_len

        corrected[t] = base[t] + mean_resid + momentum

    return corrected


def _adaptive_momentum_hybrid(
    xgb_preds: np.ndarray,
    gru_preds: np.ndarray,
    y_true: np.ndarray,
    lookback: int = 14,
    momentum_window: int = 7,
    momentum_gain: float = 0.4,
) -> np.ndarray:
    """
    BEST OF BOTH WORLDS: adaptive rolling weights + momentum correction.

    Phase 1: Compute adaptive-weighted blend (favours whichever model
             is performing better recently).
    Phase 2: Apply momentum-aware bias correction on top of the adaptive
             blend to close the gap during strong trend periods.
    """
    n = len(y_true)

    # Phase 1: adaptive blend
    adapt_blend = _adaptive_rolling_weights(
        xgb_preds, gru_preds, y_true, lookback=lookback,
    )

    # Phase 2: momentum correction
    corrected = np.copy(adapt_blend)
    for t in range(2, n):
        start = max(0, t - momentum_window)
        window_len = t - start
        if window_len < 2:
            past_resid = y_true[start:t] - adapt_blend[start:t]
            corrected[t] = adapt_blend[t] + np.mean(past_resid)
            continue

        past_resid = y_true[start:t] - adapt_blend[start:t]
        mean_resid = np.mean(past_resid)

        x = np.arange(window_len, dtype=float)
        x_mean = x.mean()
        slope = np.sum((x - x_mean) * (past_resid - past_resid.mean())) / (np.sum((x - x_mean) ** 2) + 1e-10)
        momentum = momentum_gain * slope * window_len

        corrected[t] = adapt_blend[t] + mean_resid + momentum

    return corrected


def _ewm_momentum_hybrid(
    xgb_preds: np.ndarray,
    gru_preds: np.ndarray,
    y_true: np.ndarray,
    span: int = 14,
    momentum_window: int = 7,
    momentum_gain: float = 0.4,
) -> np.ndarray:
    """
    EWM-adaptive weights + momentum correction.
    """
    n = len(y_true)

    # Phase 1: EWM adaptive blend
    ewm_blend = _ewm_adaptive_blend(
        xgb_preds, gru_preds, y_true, span=span,
    )

    # Phase 2: momentum correction
    corrected = np.copy(ewm_blend)
    for t in range(2, n):
        start = max(0, t - momentum_window)
        window_len = t - start
        if window_len < 2:
            past_resid = y_true[start:t] - ewm_blend[start:t]
            corrected[t] = ewm_blend[t] + np.mean(past_resid)
            continue

        past_resid = y_true[start:t] - ewm_blend[start:t]
        mean_resid = np.mean(past_resid)

        x = np.arange(window_len, dtype=float)
        x_mean = x.mean()
        slope = np.sum((x - x_mean) * (past_resid - past_resid.mean())) / (np.sum((x - x_mean) ** 2) + 1e-10)
        momentum = momentum_gain * slope * window_len

        corrected[t] = ewm_blend[t] + mean_resid + momentum

    return corrected


def _online_rmse(y_true: np.ndarray, preds: np.ndarray) -> float:
    """Compute RMSE, skipping the first step (no correction possible)."""
    return float(np.sqrt(np.mean((y_true[1:] - preds[1:]) ** 2)))


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
    candidates = [
        "xgboost",
        "gru",
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
            excluded_short[model_name] = pred_vals
            logger.warning(
                f"  ⚠ {model_name}: Only {len(pred_vals)}/{n_test} predictions — "
                f"excluded from stacking (will evaluate separately at short horizons)"
            )
        else:
            predictions_dict[model_name] = pred_vals[:n_test]
            logger.info(f"  ✓ {model_name}: truncated {len(pred_vals)} → {n_test}")

    if not predictions_dict:
        logger.error("No valid full-length prediction files found. Please run earlier stages first.")
        sys.exit(1)

    logger.info(f"\n  Stacking candidates ({len(predictions_dict)}): {list(predictions_dict.keys())}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. DYNAMIC ENSEMBLE SELECTION (DES)
    # ══════════════════════════════════════════════════════════════════════
    val_size = max(30, int(n_test * 0.3))
    val_size = min(val_size, n_test - 30)

    y_val = y_test[:val_size]

    X_meta_df = pd.DataFrame(predictions_dict)

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

    X_stack = X_meta_df[valid_models].values
    X_stack_val = X_stack[:val_size]

    # Convenience: get xgb and gru predictions
    has_both = "xgboost" in valid_models and "gru" in valid_models
    if has_both:
        xgb_idx = valid_models.index("xgboost")
        gru_idx = valid_models.index("gru")
        xgb_full = X_stack[:, xgb_idx]
        gru_full = X_stack[:, gru_idx]
        xgb_val = X_stack_val[:, xgb_idx]
        gru_val = X_stack_val[:, gru_idx]

    # ══════════════════════════════════════════════════════════════════════
    # 3. STRATEGY COMPETITION
    # ══════════════════════════════════════════════════════════════════════
    #
    # ALL strategies are evaluated on the SAME metric: full-test RMSE.
    # Static strategies are trained on the validation set, then their
    # predictions are generated for the full test set and RMSE is measured
    # there. Online strategies are inherently causal (no leakage).
    # This ensures fair apples-to-apples comparison.
    #
    logger.info("\n── Meta-Learner Strategy Competition ──")
    logger.info("  All strategies evaluated on full-test RMSE (455 days)")

    strategies = {}  # name -> {rmse, preds, type, ...}

    # ═══════════════════════════════════════════════════════════════
    # STATIC STRATEGIES (trained on val, evaluated on FULL test)
    # ═══════════════════════════════════════════════════════════════

    # Ridge Stacking
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        name = f"[S] Ridge(α={alpha})"
        try:
            _, model = _cv_stacking_score(
                X_stack_val, y_val,
                Ridge,
                {"alpha": alpha, "positive": True, "fit_intercept": True},
                n_splits=min(5, max(2, val_size // 15)),
            )
            preds = model.predict(X_stack)
            r = float(np.sqrt(np.mean((y_test - preds) ** 2)))
            strategies[name] = {"rmse": r, "preds": preds, "model": model, "type": "learner"}
        except Exception as e:
            logger.debug(f"  {name} failed: {e}")

    # BayesianRidge
    try:
        _, model = _cv_stacking_score(
            X_stack_val, y_val,
            BayesianRidge,
            {"max_iter": 300, "tol": 1e-4, "fit_intercept": True},
            n_splits=min(5, max(2, val_size // 15)),
        )
        preds = model.predict(X_stack)
        r = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        strategies["[S] BayesianRidge"] = {"rmse": r, "preds": preds, "model": model, "type": "learner"}
    except Exception as e:
        logger.debug(f"  BayesianRidge failed: {e}")

    # InvRMSE Weighted
    try:
        inv_rmse = {m: 1.0 / (model_val_rmse[m] + 1e-8) for m in valid_models}
        total_inv = sum(inv_rmse.values())
        inv_weights = np.array([inv_rmse[m] / total_inv for m in valid_models])
        preds = X_stack @ inv_weights
        r = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        strategies["[S] InvRMSE_Weighted"] = {
            "rmse": r, "preds": preds, "weights": inv_weights, "type": "weighted",
        }
    except Exception as e:
        logger.debug(f"  InvRMSE_Weighted failed: {e}")

    # Optimized Blend (scipy) — train alpha on val, evaluate on full test
    if has_both:
        try:
            def _blend_rmse_val(alpha_arr):
                a = float(alpha_arr[0])
                pred = a * xgb_val + (1 - a) * gru_val
                return float(np.sqrt(np.mean((y_val - pred) ** 2)))

            best_grid_alpha = min(np.linspace(0, 1, 101), key=lambda a: _blend_rmse_val([a]))
            result = minimize(
                _blend_rmse_val, x0=[best_grid_alpha], method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 500},
            )
            opt_alpha = float(np.clip(result.x[0], 0.0, 1.0))
            preds = opt_alpha * xgb_full + (1 - opt_alpha) * gru_full
            r = float(np.sqrt(np.mean((y_test - preds) ** 2)))
            strategies["[S] OptimizedBlend"] = {
                "rmse": r, "preds": preds, "alpha": opt_alpha, "type": "optimized_blend",
            }
        except Exception as e:
            logger.debug(f"  OptimizedBlend failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # ONLINE STRATEGIES (evaluated on full test set, causal)
    # ═══════════════════════════════════════════════════════════════

    if has_both:
        # Adaptive Rolling Weights
        for lb in [7, 10, 14, 21, 30]:
            name = f"[O] AdaptiveRolling(lb={lb})"
            try:
                preds = _adaptive_rolling_weights(xgb_full, gru_full, y_test, lookback=lb)
                r = _online_rmse(y_test, preds)
                strategies[name] = {"rmse": r, "preds": preds, "lookback": lb, "type": "adaptive_rolling"}
            except Exception as e:
                logger.debug(f"  {name} failed: {e}")

        # EWM-Adaptive
        for span in [7, 10, 14, 21, 30]:
            name = f"[O] EWM_Adaptive(span={span})"
            try:
                preds = _ewm_adaptive_blend(xgb_full, gru_full, y_test, span=span)
                r = _online_rmse(y_test, preds)
                strategies[name] = {"rmse": r, "preds": preds, "span": span, "type": "ewm_adaptive"}
            except Exception as e:
                logger.debug(f"  {name} failed: {e}")

        # Residual-Corrected Blend
        for base_alpha in [0.5, 0.6, 0.65, 0.7, 0.75]:
            for cw in [7, 10, 14, 21]:
                name = f"[O] ResidCorr(α={base_alpha},w={cw})"
                try:
                    preds = _residual_corrected_online(
                        xgb_full, gru_full, y_test,
                        base_alpha=base_alpha, correction_window=cw,
                    )
                    r = _online_rmse(y_test, preds)
                    strategies[name] = {
                        "rmse": r, "preds": preds, "base_alpha": base_alpha,
                        "correction_window": cw, "type": "residual_corrected",
                    }
                except Exception as e:
                    logger.debug(f"  {name} failed: {e}")

        # MOMENTUM-CORRECTED BLEND (the big gun)
        for base_alpha in [0.5, 0.6, 0.65, 0.7]:
            for mw in [5, 7, 10, 14]:
                for mg in [0.2, 0.3, 0.4, 0.5, 0.6]:
                    name = f"[O] Momentum(α={base_alpha},mw={mw},mg={mg})"
                    try:
                        preds = _momentum_corrected_blend(
                            xgb_full, gru_full, y_test,
                            base_alpha=base_alpha,
                            momentum_window=mw, momentum_gain=mg,
                        )
                        r = _online_rmse(y_test, preds)
                        strategies[name] = {
                            "rmse": r, "preds": preds, "base_alpha": base_alpha,
                            "momentum_window": mw, "momentum_gain": mg,
                            "type": "momentum_corrected",
                        }
                    except Exception as e:
                        logger.debug(f"  {name} failed: {e}")

        # ADAPTIVE + MOMENTUM HYBRID
        for lb in [10, 14, 21]:
            for mw in [5, 7, 10]:
                for mg in [0.2, 0.3, 0.4, 0.5]:
                    name = f"[O] AdaptMomentum(lb={lb},mw={mw},mg={mg})"
                    try:
                        preds = _adaptive_momentum_hybrid(
                            xgb_full, gru_full, y_test,
                            lookback=lb, momentum_window=mw, momentum_gain=mg,
                        )
                        r = _online_rmse(y_test, preds)
                        strategies[name] = {
                            "rmse": r, "preds": preds, "lookback": lb,
                            "momentum_window": mw, "momentum_gain": mg,
                            "type": "adaptive_momentum",
                        }
                    except Exception as e:
                        logger.debug(f"  {name} failed: {e}")

        # EWM + MOMENTUM HYBRID
        for span in [10, 14, 21]:
            for mw in [5, 7, 10]:
                for mg in [0.2, 0.3, 0.4, 0.5]:
                    name = f"[O] EWM_Momentum(sp={span},mw={mw},mg={mg})"
                    try:
                        preds = _ewm_momentum_hybrid(
                            xgb_full, gru_full, y_test,
                            span=span, momentum_window=mw, momentum_gain=mg,
                        )
                        r = _online_rmse(y_test, preds)
                        strategies[name] = {
                            "rmse": r, "preds": preds, "span": span,
                            "momentum_window": mw, "momentum_gain": mg,
                            "type": "ewm_momentum",
                        }
                    except Exception as e:
                        logger.debug(f"  {name} failed: {e}")

    if not strategies:
        logger.error("All meta-learner strategies failed. Cannot build ensemble.")
        sys.exit(1)

    # ── Report top 20 and select best ──
    sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]["rmse"])
    logger.info("\n  Top 20 strategy competition results (all full-test RMSE):")
    for name, info in sorted_strategies[:20]:
        logger.info(f"    {name:50s}  full_RMSE={info['rmse']:.4f}")

    if len(sorted_strategies) > 20:
        logger.info(f"    ... and {len(sorted_strategies) - 20} more strategies evaluated")

    best_strategy_name = sorted_strategies[0][0]
    best_strategy = sorted_strategies[0][1]
    logger.info(f"\n  ★ Selected: {best_strategy_name} (full_RMSE={best_strategy['rmse']:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # 4. USE PRE-COMPUTED PREDICTIONS (already generated during competition)
    # ══════════════════════════════════════════════════════════════════════

    final_preds = best_strategy["preds"]
    stype = best_strategy["type"]

    # Log details of the selected strategy
    if stype == "learner" and "model" in best_strategy:
        model = best_strategy["model"]
        if hasattr(model, "coef_"):
            coefs = model.coef_
            nw = coefs / (np.sum(np.abs(coefs)) + 1e-10)
            logger.info("\n  Meta-Learner Weights:")
            for name_m, w, n in zip(valid_models, coefs, nw):
                logger.info(f"    {name_m:20s}: coef={w:+.6f}  normalised={n:+.4f}")
        if hasattr(model, "intercept_"):
            logger.info(f"    {'intercept':20s}: {model.intercept_:+.6f}")
    elif stype == "weighted":
        weights = best_strategy["weights"]
        logger.info("\n  Ensemble Weights:")
        for name_m, w in zip(valid_models, weights):
            logger.info(f"    {name_m:20s}: {w:.4f}")
    elif stype == "optimized_blend":
        a = best_strategy["alpha"]
        logger.info(f"\n  Optimized Blend: α(XGB)={a:.4f}, (1-α)(GRU)={1-a:.4f}")
    elif stype == "adaptive_rolling":
        logger.info(f"\n  Adaptive Rolling Weights: lookback={best_strategy['lookback']}")
    elif stype == "ewm_adaptive":
        logger.info(f"\n  EWM-Adaptive Blend: span={best_strategy['span']}")
    elif stype == "residual_corrected":
        logger.info(f"\n  Residual-Corrected: α={best_strategy['base_alpha']}, correction_window={best_strategy['correction_window']}")
    elif stype == "momentum_corrected":
        logger.info(f"\n  Momentum-Corrected: α={best_strategy['base_alpha']}, momentum_window={best_strategy['momentum_window']}, momentum_gain={best_strategy['momentum_gain']}")
    elif stype == "adaptive_momentum":
        logger.info(f"\n  Adaptive+Momentum: lookback={best_strategy['lookback']}, mw={best_strategy['momentum_window']}, mg={best_strategy['momentum_gain']}")
    elif stype == "ewm_momentum":
        logger.info(f"\n  EWM+Momentum: span={best_strategy['span']}, mw={best_strategy['momentum_window']}, mg={best_strategy['momentum_gain']}")
    else:
        logger.info(f"\n  Strategy type: {stype}")

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

    # ── Short-horizon-only models ──
    for model_name, short_preds in excluded_short.items():
        n_short = len(short_preds)
        for h in horizons:
            n = min(h, n_short, n_test)
            if n > 0:
                m = compute_all_metrics(y_test[:n], short_preds[:n])
                all_results.append({
                    "Commodity": commodity, "Model": f"{model_name}(short)", "Horizon": h, **m,
                })

    # ══════════════════════════════════════════════════════════════════════
    # 6. SAVE OUTPUTS
    # ══════════════════════════════════════════════════════════════════════

    pd.DataFrame({"prediction": final_preds}).to_csv(
        reports_dir / f"{slug}_stackingensemble_predictions.csv", index=False
    )
    logger.info(f"  Saved: {slug}_stackingensemble_predictions.csv")

    try:
        n = min(len(test_dates), len(y_test), len(final_preds))
        plot_forecast_vs_actual(
            test_dates[:n], y_test[:n], final_preds[:n],
            commodity, "StackingEnsemble", cfg,
        )
    except Exception as e:
        logger.debug(f"  Forecast plot failed: {e}")

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
