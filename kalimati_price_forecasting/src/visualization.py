"""
visualization.py — Publication-Quality Plots for Kalimati Price Forecasting
============================================================================

Generates high-resolution figures for:
    • Time-series trends & decomposition
    • ACF / PACF correlograms
    • Seasonal boxplots (monthly, day-of-week)
    • Festival-adjusted heatmaps
    • Forecast vs. actual overlays with prediction intervals
    • Residual diagnostics
    • Feature importance (bar + SHAP)
    • Model comparison tables
    • Training history (DL loss curves)

All figures are saved at 300 DPI (publication-ready).

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("kalimati.visualization")

# ── Global Style ──
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 300,
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("Set2", 10)


def _save_fig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def _fig_dir(cfg: Dict[str, Any]) -> Path:
    d = Path(cfg["output"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 1. TIME-SERIES PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_price_series(
    df: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
    target: str = "Average",
) -> None:
    """Plot full price history with min/max band."""
    fig, ax = plt.subplots(figsize=(16, 6))
    sub = df[df["Commodity"] == commodity].sort_values("Date")

    ax.plot(sub["Date"], sub[target], color=PALETTE[0], lw=1.2, label="Average")
    ax.fill_between(sub["Date"], sub["Minimum"], sub["Maximum"],
                     alpha=0.2, color=PALETTE[0], label="Min–Max band")
    ax.set_title(f"{commodity} — Daily Price (NPR/Kg)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (NPR)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    from src.utils import sanitize_commodity_name
    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_price_series.png")


def plot_multi_commodity(
    df: pd.DataFrame, commodities: List[str], cfg: Dict[str, Any],
    target: str = "Average",
) -> None:
    """Overlay multiple commodities on one chart."""
    fig, ax = plt.subplots(figsize=(16, 7))
    for i, c in enumerate(commodities):
        sub = df[df["Commodity"] == c].sort_values("Date")
        ax.plot(sub["Date"], sub[target], lw=1.0, label=c, color=PALETTE[i % 10])
    ax.set_title("Daily Average Prices — Selected Commodities", fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (NPR)")
    ax.legend(loc="upper left", ncol=2)
    _save_fig(fig, _fig_dir(cfg) / "multi_commodity_prices.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. SEASONAL DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

def plot_decomposition(
    df: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
    target: str = "Average", period: int = 365,
) -> None:
    """STL decomposition plot."""
    from statsmodels.tsa.seasonal import STL
    from src.utils import sanitize_commodity_name

    sub = df[df["Commodity"] == commodity].sort_values("Date").set_index("Date")
    series = sub[target].dropna()
    if len(series) < 2 * period:
        period = 7

    stl = STL(series, period=period, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    for ax, data, title in zip(
        axes,
        [series, res.trend, res.seasonal, res.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
    ):
        ax.plot(data.index, data.values, lw=0.8, color=PALETTE[0])
        ax.set_ylabel(title); ax.set_title(title, fontsize=12)

    fig.suptitle(f"{commodity} — STL Decomposition", fontweight="bold", y=1.01)
    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ACF / PACF
# ═══════════════════════════════════════════════════════════════════════════

def plot_acf_pacf(
    df: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
    target: str = "Average", lags: int = 60,
) -> None:
    """ACF and PACF correlograms."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from src.utils import sanitize_commodity_name

    sub = df[df["Commodity"] == commodity].sort_values("Date")
    series = sub[target].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series, lags=lags, ax=ax1, alpha=0.05)
    ax1.set_title(f"{commodity} — ACF")
    plot_pacf(series, lags=lags, ax=ax2, alpha=0.05, method="ywm")
    ax2.set_title(f"{commodity} — PACF")

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_acf_pacf.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SEASONAL BOXPLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_seasonal_boxplots(
    df: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
    target: str = "Average",
) -> None:
    """Monthly and day-of-week boxplots."""
    from src.utils import sanitize_commodity_name
    sub = df[df["Commodity"] == commodity].copy()
    sub["Month"] = pd.to_datetime(sub["Date"]).dt.month
    sub["DayOfWeek"] = pd.to_datetime(sub["Date"]).dt.day_name()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    sns.boxplot(x="Month", y=target, data=sub, ax=ax1, palette="coolwarm",
                fliersize=2)
    ax1.set_title(f"{commodity} — Monthly Distribution")

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    sns.boxplot(x="DayOfWeek", y=target, data=sub, ax=ax2, palette="Set2",
                order=day_order, fliersize=2)
    ax2.set_title(f"{commodity} — Day-of-Week Distribution")
    ax2.tick_params(axis="x", rotation=45)

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_seasonal_boxplots.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. FESTIVAL HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_festival_heatmap(
    df: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
    target: str = "Average",
) -> None:
    """Heatmap of average prices during festival vs non-festival periods."""
    from src.utils import sanitize_commodity_name
    sub = df[df["Commodity"] == commodity].copy()
    fest_cols = [c for c in sub.columns if c.startswith("fest_") and c != "fest_any"]
    if not fest_cols:
        return

    sub["Year"] = pd.to_datetime(sub["Date"]).dt.year
    records = []
    for fc in fest_cols:
        fname = fc.replace("fest_", "").replace("_", " ").title()
        for year in sub["Year"].unique():
            mask = (sub["Year"] == year) & (sub[fc] == 1)
            if mask.sum() > 0:
                records.append({"Festival": fname, "Year": year,
                                "AvgPrice": sub.loc[mask, target].mean()})
    if not records:
        return

    pivot = pd.DataFrame(records).pivot_table(
        index="Festival", columns="Year", values="AvgPrice"
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title(f"{commodity} — Festival Period Average Prices (NPR)", fontweight="bold")

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_festival_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. FORECAST PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_forecast_vs_actual(
    dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
    commodity: str, model_name: str, cfg: Dict[str, Any],
    lower: Optional[np.ndarray] = None, upper: Optional[np.ndarray] = None,
) -> None:
    """Forecast vs actual with optional prediction intervals."""
    from src.utils import sanitize_commodity_name
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, y_true, label="Actual", color=PALETTE[0], lw=1.2)
    ax.plot(dates[:len(y_pred)], y_pred, label=f"{model_name} Forecast",
            color=PALETTE[1], lw=1.2, ls="--")
    if lower is not None and upper is not None:
        ax.fill_between(dates[:len(lower)], lower, upper,
                         alpha=0.15, color=PALETTE[1], label="95% PI")
    ax.set_title(f"{commodity} — {model_name} Forecast vs Actual", fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (NPR)")
    ax.legend()

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_{model_name.lower()}_forecast.png")


def plot_all_models_forecast(
    dates: np.ndarray, y_true: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    commodity: str, cfg: Dict[str, Any],
) -> None:
    """Overlay forecasts from all models."""
    from src.utils import sanitize_commodity_name
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(dates, y_true, label="Actual", color="black", lw=1.5)
    for i, (name, pred) in enumerate(model_preds.items()):
        n = min(len(dates), len(pred))
        ax.plot(dates[:n], pred[:n], label=name, lw=1.0,
                ls="--", color=PALETTE[i % 10])
    ax.set_title(f"{commodity} — All Models Comparison", fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (NPR)")
    ax.legend(loc="upper left", ncol=2)

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_all_models_forecast.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. RESIDUAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_diagnostics(
    residuals: np.ndarray, model_name: str, commodity: str,
    cfg: Dict[str, Any],
) -> None:
    """4-panel residual diagnostic plot."""
    from statsmodels.graphics.tsaplots import plot_acf
    from src.utils import sanitize_commodity_name

    residuals = residuals[~np.isnan(residuals)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series of residuals
    axes[0, 0].plot(residuals, lw=0.6, color=PALETTE[0])
    axes[0, 0].axhline(0, color="red", ls="--", lw=0.8)
    axes[0, 0].set_title("Residuals over Time")

    # Histogram
    axes[0, 1].hist(residuals, bins=50, edgecolor="white", color=PALETTE[1], density=True)
    axes[0, 1].set_title("Residual Distribution")

    # Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # ACF
    plot_acf(residuals, lags=30, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title("Residual ACF")

    fig.suptitle(f"{commodity} — {model_name} Residual Diagnostics",
                 fontweight="bold", y=1.02)

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_{model_name.lower()}_residuals.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    importance_df: pd.DataFrame, model_name: str, commodity: str,
    cfg: Dict[str, Any], top_n: int = 20,
) -> None:
    """Horizontal bar chart of feature importances."""
    from src.utils import sanitize_commodity_name
    top = importance_df.head(top_n).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["Feature"], top["Importance"], color=PALETTE[2])
    ax.set_title(f"{commodity} — {model_name} Feature Importance",
                 fontweight="bold")
    ax.set_xlabel("Importance")

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_{model_name.lower()}_importance.png")


def plot_shap_summary(
    model: Any, X: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
    model_name: str = "XGBoost", max_samples: int = 500,
) -> None:
    """SHAP summary plot for tree-based models."""
    from src.utils import sanitize_commodity_name
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        X_sample = X.sample(min(max_samples, len(X)), random_state=42)
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        slug = sanitize_commodity_name(commodity)
        plt.savefig(str(_fig_dir(cfg) / f"{slug}_{model_name.lower()}_shap.png"),
                     dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP plot saved for {commodity}")
    except Exception as e:
        logger.warning(f"SHAP plot failed for {commodity}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. DL TRAINING HISTORY
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_history(
    history: Any, model_name: str, commodity: str, cfg: Dict[str, Any],
) -> None:
    """Plot training and validation loss curves.

    Handles both PyTorch dict format and Keras history objects.
    """
    from src.utils import sanitize_commodity_name

    # Handle both PyTorch dict and Keras history object
    if isinstance(history, dict):
        train_loss = history.get("train_loss", history.get("loss", []))
        val_loss = history.get("val_loss", [])
    else:
        train_loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])

    if not train_loss:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Train Loss", color=PALETTE[0])
    if val_loss:
        ax.plot(epochs, val_loss, label="Val Loss", color=PALETTE[1])
    ax.set_title(f"{commodity} — {model_name} Training History", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)")
    ax.legend()

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_{model_name.lower()}_history.png")


# ═══════════════════════════════════════════════════════════════════════════
# 10. MODEL COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    results_df: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
    metric: str = "RMSE", horizon: int = 30,
) -> None:
    """Grouped bar chart comparing models for a given commodity/horizon."""
    from src.utils import sanitize_commodity_name
    mask = (results_df["Commodity"] == commodity)
    if "Horizon" in results_df.columns:
        mask = mask & (results_df["Horizon"] == horizon)
    subset = results_df[mask].sort_values(metric)

    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PALETTE[i % 10] for i in range(len(subset))]
    bars = ax.bar(subset["Model"], subset[metric], color=colors, edgecolor="white")

    for bar, val in zip(bars, subset[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{commodity} — Model Comparison ({metric}, h={horizon})",
                 fontweight="bold")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)

    slug = sanitize_commodity_name(commodity)
    _save_fig(fig, _fig_dir(cfg) / f"{slug}_model_comparison_{metric.lower()}_h{horizon}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 11. MASTER EDA FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_eda_plots(
    df: pd.DataFrame, commodity: str, cfg: Dict[str, Any],
) -> None:
    """Generate the full EDA suite for one commodity."""
    logger.info(f"Generating EDA plots for {commodity} …")
    plot_price_series(df, commodity, cfg)
    plot_decomposition(df, commodity, cfg)
    plot_acf_pacf(df, commodity, cfg)
    plot_seasonal_boxplots(df, commodity, cfg)
    plot_festival_heatmap(df, commodity, cfg)
    logger.info(f"EDA plots complete for {commodity}")
