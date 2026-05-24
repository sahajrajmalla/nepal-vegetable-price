"""
statistical.py — ARIMA / SARIMA Statistical Forecasting Models
===============================================================

Implements:
    1. Auto-ARIMA via pmdarima (automatic order selection).
    2. Manual SARIMA via statsmodels (for when orders are specified).
    3. Recursive multi-step forecasting.
    4. In-sample residual extraction for hybrid models.
    5. Confidence interval generation (built-in).

These models capture linear autocorrelation and seasonal patterns.
They serve as strong benchmarks and as components of hybrid models.

References
----------
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015).
  Time Series Analysis: Forecasting and Control. Wiley.
- Hyndman, R.J. & Khandakar, Y. (2008). Automatic Time Series
  Forecasting: The forecast Package for R. Journal of Statistical
  Software, 27(3).

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

logger = logging.getLogger("kalimati.models.statistical")


def fit_auto_arima(
    train_y: np.ndarray,
    cfg: Dict[str, Any],
    exog_train: Optional[np.ndarray] = None,
) -> Any:
    """
    Fit an Auto-ARIMA model using pmdarima's stepwise algorithm.

    Parameters
    ----------
    train_y : array-like
        Training target series.
    cfg : dict
        Pipeline configuration (ARIMA hyperparameters).
    exog_train : array-like, optional
        Exogenous regressors for the training period.

    Returns
    -------
    pmdarima.ARIMA
        Fitted auto-ARIMA model.
    """
    import pmdarima as pm

    arima_cfg = cfg.get("models", {}).get("statistical", {}).get("arima", {})
    m = arima_cfg.get("m", 7)
    seasonal = arima_cfg.get("seasonal", True)
    stepwise = arima_cfg.get("stepwise", True)
    ic = arima_cfg.get("information_criterion", "aic")
    trace = arima_cfg.get("trace", False)

    logger.info(f"Fitting Auto-ARIMA (m={m}, seasonal={seasonal}, IC={ic}) …")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            train_y,
            exogenous=exog_train,
            seasonal=seasonal,
            m=m,
            max_p=arima_cfg.get("max_p", 5),
            max_q=arima_cfg.get("max_q", 5),
            max_d=arima_cfg.get("max_d", 2),
            max_P=arima_cfg.get("max_P", 2),
            max_Q=arima_cfg.get("max_Q", 2),
            max_D=arima_cfg.get("max_D", 1),
            stepwise=stepwise,
            information_criterion=ic,
            trace=trace,
            error_action="ignore",
            suppress_warnings=True,
            n_fits=50,
        )

    order = model.order
    seasonal_order = model.seasonal_order
    aic = model.aic()
    logger.info(
        f"Auto-ARIMA selected: ARIMA{order} × {seasonal_order}  "
        f"AIC={aic:.2f}"
    )
    return model


def predict_arima(
    model: Any,
    horizon: int,
    exog_test: Optional[np.ndarray] = None,
    return_ci: bool = True,
    confidence: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Generate forecasts from a fitted ARIMA/SARIMA model.

    Parameters
    ----------
    model : pmdarima.ARIMA
        Fitted ARIMA model.
    horizon : int
        Number of periods to forecast.
    exog_test : array-like, optional
        Exogenous regressors for the forecast period.
    return_ci : bool
        Whether to return confidence intervals.
    confidence : float
        Confidence level for prediction intervals.

    Returns
    -------
    dict
        'predictions': point forecasts (np.ndarray)
        'lower': lower CI bound (if return_ci)
        'upper': upper CI bound (if return_ci)
    """
    alpha = 1 - confidence

    if return_ci:
        forecast, ci = model.predict(
            n_periods=horizon,
            exogenous=exog_test,
            return_conf_int=True,
            alpha=alpha,
        )
        return {
            "predictions": np.asarray(forecast),
            "lower": ci[:, 0],
            "upper": ci[:, 1],
        }
    else:
        forecast = model.predict(
            n_periods=horizon,
            exogenous=exog_test,
            return_conf_int=False,
        )
        return {"predictions": np.asarray(forecast)}


def get_arima_residuals(model: Any) -> np.ndarray:
    """
    Extract in-sample residuals from a fitted ARIMA model.

    Parameters
    ----------
    model : pmdarima.ARIMA
        Fitted model.

    Returns
    -------
    np.ndarray
        Residual array.
    """
    return np.asarray(model.resid())


def fit_sarima_manual(
    train_y: np.ndarray,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
    exog_train: Optional[np.ndarray] = None,
) -> Any:
    """
    Fit a manual SARIMA model via statsmodels.

    Parameters
    ----------
    train_y : array-like
        Training series.
    order : tuple
        (p, d, q) ARIMA order.
    seasonal_order : tuple
        (P, D, Q, m) seasonal ARIMA order.
    exog_train : array-like, optional
        Exogenous regressors.

    Returns
    -------
    statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        Fitted SARIMAX model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    logger.info(
        f"Fitting manual SARIMA{order} × {seasonal_order} …"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_y,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)

    logger.info(
        f"SARIMA fitted — AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}"
    )
    return fitted


def predict_sarima_manual(
    model: Any,
    horizon: int,
    exog_test: Optional[np.ndarray] = None,
    confidence: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Forecast from a statsmodels SARIMAX fit.

    Parameters
    ----------
    model : SARIMAXResultsWrapper
        Fitted model.
    horizon : int
        Forecast horizon.
    exog_test : array-like, optional
        Exogenous variables.
    confidence : float
        CI confidence level.

    Returns
    -------
    dict
        'predictions', 'lower', 'upper'.
    """
    alpha = 1 - confidence
    forecast = model.get_forecast(steps=horizon, exog=exog_test)

    # Handle both pandas Series and numpy array returns
    pred_mean_raw = forecast.predicted_mean
    pred_mean = np.asarray(pred_mean_raw).flatten()

    ci_raw = forecast.conf_int(alpha=alpha)
    if hasattr(ci_raw, 'iloc'):
        lower = ci_raw.iloc[:, 0].values
        upper = ci_raw.iloc[:, 1].values
    else:
        ci_arr = np.asarray(ci_raw)
        lower = ci_arr[:, 0]
        upper = ci_arr[:, 1]

    return {
        "predictions": pred_mean,
        "lower": lower,
        "upper": upper,
    }


def run_statistical_models(
    train_y: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Run all configured statistical models.

    Predictions are always of length ``len(test_y)`` so that
    ``run_all.py`` can evaluate at any horizon by slicing.

    Parameters
    ----------
    train_y : array-like
        Training target.
    test_y : array-like
        Test target.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Model name → {predictions, lower, upper, metrics, model, residuals}.
    """
    from src.evaluation import compute_all_metrics

    train_y = np.asarray(train_y, dtype=float)
    test_y = np.asarray(test_y, dtype=float)
    h = len(test_y)  # Always forecast the full test set
    metrics_list = cfg.get("evaluation", {}).get("metrics", ["RMSE", "MAE", "MAPE", "sMAPE"])
    stat_cfg = cfg.get("models", {}).get("statistical", {})
    results = {}

    # Auto-ARIMA
    if stat_cfg.get("arima", {}).get("enabled", True):
        try:
            model = fit_auto_arima(train_y, cfg)
            output = predict_arima(model, h)
            pred = output["predictions"]
            metrics = compute_all_metrics(test_y, pred, metrics_list)
            residuals = get_arima_residuals(model)

            results["Auto_ARIMA"] = {
                "predictions": pred,
                "lower": output.get("lower"),
                "upper": output.get("upper"),
                "metrics": metrics,
                "model": model,
                "residuals": residuals,
            }
            logger.info(f"Auto-ARIMA — RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"Auto-ARIMA failed: {e}")

    # Manual SARIMA (if configured)
    if stat_cfg.get("sarima", {}).get("enabled", True):
        try:
            order = stat_cfg["sarima"].get("order")
            seasonal_order = stat_cfg["sarima"].get("seasonal_order")

            if order is None or seasonal_order is None:
                # Use auto-selected orders as fallback
                if "Auto_ARIMA" in results:
                    auto_model = results["Auto_ARIMA"]["model"]
                    order = auto_model.order
                    seasonal_order = auto_model.seasonal_order
                else:
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, 7)

            order = tuple(order)
            seasonal_order = tuple(seasonal_order)

            model = fit_sarima_manual(train_y, order, seasonal_order)
            output = predict_sarima_manual(model, h)
            pred = output["predictions"]
            metrics = compute_all_metrics(test_y, pred, metrics_list)
            residuals = train_y - np.asarray(model.fittedvalues)

            results["SARIMA"] = {
                "predictions": pred,
                "lower": output.get("lower"),
                "upper": output.get("upper"),
                "metrics": metrics,
                "model": model,
                "residuals": residuals,
            }
            logger.info(f"SARIMA — RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"SARIMA failed: {e}")

    return results


def save_statistical_model(
    model: Any,
    cfg: Dict[str, Any],
    commodity: str,
    model_name: str,
) -> Path:
    """
    Serialize a statistical model to disk.

    Parameters
    ----------
    model : fitted model
        ARIMA or SARIMAX model.
    cfg : dict
        Pipeline configuration.
    commodity : str
        Commodity name.
    model_name : str
        Model identifier (e.g., 'auto_arima').

    Returns
    -------
    Path
        Path to saved model file.
    """
    from src.utils import sanitize_commodity_name

    out_dir = Path(cfg["output"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_commodity_name(commodity)
    path = out_dir / f"{slug}_{model_name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Saved {model_name} for {commodity}: {path}")
    return path
