"""
ml_models.py — Machine Learning Forecasting Models
====================================================

Implements tree-based ML models for time-series regression:
    1. RandomForestRegressor
    2. ExtraTreesRegressor
    3. HistGradientBoostingRegressor (scikit-learn's native LightGBM-equivalent)
    4. XGBRegressor

All models:
    • Use TimeSeriesSplit for proper temporal cross-validation
    • Use Optuna TPE (Bayesian) hyperparameter optimisation
    • Support recursive multi-step forecasting
    • Use n_jobs=1/nthread=1 to avoid macOS ARM64 multiprocessing segfaults

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import gc
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

logger = logging.getLogger("kalimati.models.ml")

# Silence Optuna's internal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress all warnings in this module
warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# 1. RANDOM FOREST
# ═════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """Train a Random Forest with Optuna Bayesian hyperparameter optimisation."""
    rf_cfg = cfg.get("models", {}).get("ml", {}).get("random_forest", {})
    n_trials = rf_cfg.get("n_iter_search", 50)
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        model = RandomForestRegressor(**params, random_state=seed, n_jobs=1)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
        return -scores.mean()

    logger.info(f"Random Forest Optuna search: {n_trials} trials, 5-fold TimeSeriesSplit")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_mse = study.best_value

    best_model = RandomForestRegressor(**best_params, random_state=seed, n_jobs=1)
    best_model.fit(X_train, y_train)

    logger.info(f"Random Forest best params: {best_params}\n  Best CV MSE: {best_mse:.4f} (RMSE: {np.sqrt(best_mse):.4f})")
    del study
    gc.collect()
    return best_model, best_params


def get_rf_feature_importance(model: RandomForestRegressor, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Extract and rank feature importances from Random Forest."""
    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ═════════════════════════════════════════════════════════════════════════════
# 2. EXTRA TREES
# ═════════════════════════════════════════════════════════════════════════════

def train_extra_trees(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[ExtraTreesRegressor, Dict[str, Any]]:
    """Train an Extra Trees Regressor with Optuna Bayesian hyperparameter optimisation."""
    et_cfg = cfg.get("models", {}).get("ml", {}).get("extra_trees", {})
    n_trials = et_cfg.get("n_iter_search", 50)
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        model = ExtraTreesRegressor(**params, random_state=seed, n_jobs=1)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
        return -scores.mean()

    logger.info(f"Extra Trees Optuna search: {n_trials} trials, 5-fold TimeSeriesSplit")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_mse = study.best_value

    best_model = ExtraTreesRegressor(**best_params, random_state=seed, n_jobs=1)
    best_model.fit(X_train, y_train)

    logger.info(f"Extra Trees best params: {best_params}\n  Best CV MSE: {best_mse:.4f} (RMSE: {np.sqrt(best_mse):.4f})")
    del study
    gc.collect()
    return best_model, best_params


def get_et_feature_importance(model: ExtraTreesRegressor, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Extract feature importances from Extra Trees."""
    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ═════════════════════════════════════════════════════════════════════════════
# 3. HIST GRADIENT BOOSTING
# ═════════════════════════════════════════════════════════════════════════════

def train_hist_gb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, Any]]:
    """Train a HistGradientBoostingRegressor with Optuna Bayesian hyperparameter optimisation."""
    hg_cfg = cfg.get("models", {}).get("ml", {}).get("hist_gb", {})
    n_trials = hg_cfg.get("n_iter_search", 50)
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
        model = HistGradientBoostingRegressor(**params, random_state=seed)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
        return -scores.mean()

    logger.info(f"Hist Gradient Boosting Optuna search: {n_trials} trials, 5-fold TimeSeriesSplit")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_mse = study.best_value

    best_model = HistGradientBoostingRegressor(**best_params, random_state=seed)
    best_model.fit(X_train, y_train)

    logger.info(f"Hist Gradient Boosting best params: {best_params}\n  Best CV MSE: {best_mse:.4f} (RMSE: {np.sqrt(best_mse):.4f})")
    del study
    gc.collect()
    return best_model, best_params


def get_hg_feature_importance(model: HistGradientBoostingRegressor, X_train: pd.DataFrame, y_train: pd.Series, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Extract feature importances from HistGradientBoosting using permutation importance."""
    n_samples = min(300, len(X_train))
    sample_idx = X_train.sample(n_samples, random_state=42).index
    res = permutation_importance(model, X_train.loc[sample_idx], y_train.loc[sample_idx], n_repeats=3, random_state=42, n_jobs=1)
    importances = res.importances_mean
    # clip at 0
    importances = np.clip(importances, 0, None)
    # normalise
    total = importances.sum()
    if total > 0:
        importances /= total
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ═════════════════════════════════════════════════════════════════════════════
# 4. XGBOOST
# ═════════════════════════════════════════════════════════════════════════════

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an XGBoost regressor with Optuna Bayesian optimisation (n_jobs=1)."""
    import xgboost as xgb

    xgb_cfg = cfg.get("models", {}).get("ml", {}).get("xgboost", {})
    n_trials = xgb_cfg.get("n_iter_search", 80)
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        model = xgb.XGBRegressor(**params, random_state=seed, objective="reg:squarederror", n_jobs=1, nthread=1, verbosity=0)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
        return -scores.mean()

    logger.info(f"XGBoost Optuna search: {n_trials} trials, 5-fold TimeSeriesSplit")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_mse = study.best_value

    best_model = xgb.XGBRegressor(**best_params, random_state=seed, objective="reg:squarederror", n_jobs=1, nthread=1, verbosity=0)
    best_model.fit(X_train, y_train)

    logger.info(f"XGBoost best params: {best_params}\n  Best CV MSE: {best_mse:.4f} (RMSE: {np.sqrt(best_mse):.4f})")
    del study
    gc.collect()
    return best_model, best_params


def get_xgb_feature_importance(model: Any, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Extract feature importances from XGBoost."""
    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ═════════════════════════════════════════════════════════════════════════════
# 5. RECURSIVE FORECASTER
# ═════════════════════════════════════════════════════════════════════════════

def predict_recursive_ml(
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Generate recursive multi-step forecasts for any tree-based model.
    Avoids lookahead bias by re-calculating lag/rolling features at each step
    using previous model predictions.
    """
    from src.feature_engineering import engineer_features

    # Work on copies to prevent modifying caller's DataFrames
    train_df = train_df.copy()
    test_df = test_df.copy()

    target = cfg["preprocessing"]["target_column"]
    base_cols = ["Date", "Commodity", "Unit", "Minimum", "Maximum", target]

    for col in base_cols:
        if col not in train_df.columns:
            train_df[col] = "KVPI" if col == "Commodity" else ("Index" if col == "Unit" else train_df[target] if col in ["Minimum", "Maximum"] else None)
        if col not in test_df.columns:
            test_df[col] = "KVPI" if col == "Commodity" else ("Index" if col == "Unit" else test_df[target] if col in ["Minimum", "Maximum"] else None)

    train_base = train_df[base_cols].copy()
    test_base = test_df[base_cols].copy()

    # Combine train and test base target values
    combined = pd.concat([train_base, test_base], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    n_train = len(train_df)
    n_test = len(test_df)

    # Fill test portion of target with NaN to avoid any lookahead contamination
    combined.loc[n_train:, target] = np.nan
    predictions = []

    for i in range(n_test):
        idx = n_train + i
        # Keep only history up to current step
        sub_df = combined.iloc[:idx + 1].copy()

        # Recompute features on the updated history
        sub_df_feat = engineer_features(sub_df, cfg, commodity="KVPI")

        # Get features for current prediction step (the last row)
        X_curr = sub_df_feat.iloc[-1:].reindex(columns=feature_cols, fill_value=0)

        # Predict
        pred_val = model.predict(X_curr)[0]
        predictions.append(pred_val)

        # Update target in history for future lags/rolling stats
        combined.loc[idx, target] = pred_val

    return np.array(predictions)


# ═════════════════════════════════════════════════════════════════════════════
# 6. UNIFIED ML RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_ml_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: Dict[str, Any],
    feature_cols: List[str],
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate machine learning models (Random Forest, Extra Trees, HistGB, XGBoost)
    with Bayesian hyperparameter optimization and recursive forecasting.
    """
    ml_cfg = cfg.get("models", {}).get("ml", {})
    strategy = cfg["evaluation"].get("strategy", "recursive")
    logger.info(f"ML Model Tuning Strategy: {strategy.upper()}")

    models_to_run = []
    if ml_cfg.get("random_forest", {}).get("enabled", True):
        models_to_run.append(("RandomForest", train_random_forest, get_rf_feature_importance))
    if ml_cfg.get("extra_trees", {}).get("enabled", True):
        models_to_run.append(("ExtraTrees", train_extra_trees, get_et_feature_importance))
    if ml_cfg.get("hist_gb", {}).get("enabled", True):
        models_to_run.append(("HistGB", train_hist_gb, lambda m, f: get_hg_feature_importance(m, X_train, y_train, f)))
    if ml_cfg.get("xgboost", {}).get("enabled", True):
        models_to_run.append(("XGBoost", train_xgboost, get_xgb_feature_importance))

    results = {}
    for name, train_fn, importance_fn in models_to_run:
        try:
            logger.info(f"\n── Training {name} ──")
            # 1. Fit model with tuning
            model, params = train_fn(X_train, y_train, cfg, seed)

            # 2. Forecast
            if strategy == "recursive":
                logger.info(f"Generating recursive out-of-sample forecast for {name}…")
                pred = predict_recursive_ml(model, train_df, test_df, feature_cols, cfg)
            else:
                logger.info(f"Generating one-step-ahead rolling forecast for {name}…")
                pred = model.predict(X_test)

            # 3. Compile results
            res = {
                "predictions": pred,
                "model": model,
            }

            # 4. Feature importance
            if importance_fn is not None:
                try:
                    res["feature_importance"] = importance_fn(model, feature_cols)
                except Exception as e:
                    logger.warning(f"Failed to calculate feature importance for {name}: {e}")

            results[name] = res

        except Exception as e:
            logger.error(f"Failed to run model {name}: {e}", exc_info=True)

    return results


def save_ml_model(
    model: Any,
    cfg: Dict[str, Any],
    commodity: str,
    model_name: str,
) -> Path:
    """Serialize an ML model to disk via joblib."""
    from src.utils import sanitize_commodity_name

    out_dir = Path(cfg["output"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_commodity_name(commodity)
    path = out_dir / f"{slug}_{model_name}.joblib"
    joblib.dump(model, path)
    logger.info(f"Saved {model_name} for {commodity}: {path}")
    return path
