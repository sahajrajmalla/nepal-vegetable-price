# Kalimati Vegetable Price Forecasting

> **Multi-Model Time-Series Forecasting of Daily Wholesale Vegetable Prices at Nepal's Kalimati Fruits & Vegetables Market (2013–2023)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Data Setup](#data-setup)
5. [Quick Start](#quick-start)
6. [Pipeline Architecture](#pipeline-architecture)
7. [Models](#models)
8. [Configuration](#configuration)
9. [Outputs](#outputs)
10. [Testing](#testing)
11. [Results Summary](#results-summary)
12. [Citation](#citation)
13. [References](#references)

---

## Overview

This project implements a comprehensive forecasting pipeline for predicting daily vegetable prices at the **Kalimati Fruits and Vegetables Market**, Nepal's largest wholesale market located in Kathmandu. The pipeline compares **7 models** across **4 forecast horizons** (7, 14, 30, 90 days) for **8 key commodities**.

### Key Features

- **Data Pipeline**: Automated ingestion of heterogeneous CSVs, outlier detection (IQR + domain rules), intelligent imputation, and Parquet export
- **Rich Feature Engineering**: 50+ features including autoregressive lags, rolling statistics, EWMA, cyclical calendar encoding, Nepali festival dummies, and price-derived features
- **7 Forecasting Models**: Naïve, Seasonal Naïve, Auto-ARIMA, SARIMA, Random Forest, XGBoost, LSTM, GRU, ARIMA-LSTM hybrid, ARIMA-XGBoost hybrid
- **Rigorous Evaluation**: Time-series cross-validation, multi-horizon metrics (RMSE, MAE, MAPE, sMAPE), Diebold-Mariano test, residual diagnostics
- **Explainability**: SHAP values for tree models, feature importance rankings, festival impact analysis
- **Publication-Quality Outputs**: 300 DPI figures, Markdown/CSV result tables, serialized models

---

## Project Structure

```
kalimati_price_forecasting/
├── configs/
│   └── default.yaml          # Master YAML configuration
├── data/
│   └── raw/                  # Place raw CSVs here
├── notebooks/                # EDA Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Ingestion, cleaning, imputation
│   ├── feature_engineering.py# Lags, rolling, calendar, festivals
│   ├── evaluation.py         # Metrics, CV, Diebold-Mariano
│   ├── utils.py              # Config, logging, seeds, helpers
│   ├── visualization.py      # Publication-quality plots
│   └── models/
│       ├── __init__.py
│       ├── baselines.py      # Naïve, Seasonal Naïve
│       ├── statistical.py    # Auto-ARIMA, SARIMA
│       ├── ml_models.py      # Random Forest, XGBoost
│       ├── dl_models.py      # LSTM, GRU
│       └── hybrid.py         # ARIMA-LSTM, ARIMA-XGBoost
├── outputs/
│   ├── cleaned_data/         # Parquet + CSV exports
│   ├── figures/              # All generated plots
│   ├── models/               # Serialized models
│   └── reports/              # CSV tables, Markdown summary
├── tests/
│   └── test_pipeline.py      # Unit tests
├── requirements.txt
├── run_all.py                # Master pipeline script
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sahajrajmalla/nepal-vegetable-price.git
cd nepal-vegetable-price/kalimati_price_forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Python**: 3.9 – 3.11 recommended. TensorFlow 2.12+ required for DL models.

---

## Data Setup

Place raw CSV files in `data/raw/`:

```
data/raw/
├── Kalimati Tarkari Prices from June 2013 to May 2021.csv
└── Kalimati Tarkari Prices from May 2021 to September 2023.csv
```

**Expected schema**: `SN, Commodity, Date, Unit, Minimum, Maximum, Average`

The pipeline handles schema differences between files automatically (date formats, currency prefixes, missing columns).

### Data Sources

- [Kaggle: Kalimati Tarkari Dataset](https://www.kaggle.com/datasets/saurabhshahane/kalimati-tarkari-dataset)
- [Open Data Nepal](https://opendatanepal.com/)

---

## Quick Start

```bash
# Run full pipeline for all configured commodities
python run_all.py

# Run for a single commodity
python run_all.py --commodity "Tomato Big(Nepali)"

# Skip deep learning (faster)
python run_all.py --skip-dl

# Custom horizons
python run_all.py --horizons 7,14,30

# Skip EDA plots
python run_all.py --skip-eda
```

---

## Pipeline Architecture

```
Raw CSVs → Preprocessing → Feature Engineering → Train/Test Split
                                                       │
                    ┌──────────────────────────────────┤
                    │              │              │     │
                Baselines    Statistical     ML Models  DL Models
                (Naïve,      (ARIMA,        (RF,       (LSTM,
                 S-Naïve)     SARIMA)        XGBoost)   GRU)
                    │              │              │     │
                    │         Hybrid Models ←─────┘────┘
                    │         (ARIMA-LSTM, ARIMA-XGB)
                    └──────────────────────────────────┤
                                                       │
                              Evaluation & Comparison
                              (Metrics, CV, DM Test)
                                       │
                              Outputs (Figures, Tables,
                              Models, Reports)
```

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| Naïve | Baseline | Last-value forecast (random walk) |
| Seasonal Naïve | Baseline | Same-day-last-week forecast |
| Auto-ARIMA | Statistical | Automatic order selection via pmdarima |
| SARIMA | Statistical | Seasonal ARIMA (weekly seasonality) |
| Random Forest | ML | Bagged decision trees with TimeSeriesSplit tuning |
| XGBoost | ML | Gradient-boosted trees with regularisation |
| LSTM | Deep Learning | 2-layer Long Short-Term Memory network |
| GRU | Deep Learning | 2-layer Gated Recurrent Unit network |
| ARIMA-LSTM | Hybrid | ARIMA + LSTM on residuals (Zhang, 2003) |
| ARIMA-XGBoost | Hybrid | ARIMA fitted values as XGBoost feature |

---

## Configuration

All parameters are controlled via `configs/default.yaml`:

- **Commodities**: Add/remove from `commodities.selected`
- **Features**: Toggle lags, rolling windows, calendar features
- **Models**: Enable/disable models, set hyperparameter search spaces
- **Evaluation**: Train/test split dates, CV folds, horizons
- **Festivals**: Nepali festival dates with lead/lag impact windows

---

## Outputs

| Directory | Contents |
|-----------|----------|
| `outputs/cleaned_data/` | Parquet + CSV per commodity |
| `outputs/figures/` | Price series, decomposition, ACF/PACF, seasonal boxplots, festival heatmaps, forecast plots, residual diagnostics, SHAP, feature importance, model comparisons |
| `outputs/models/` | Serialized models (joblib/keras) |
| `outputs/reports/` | `model_comparison.csv`, `model_comparison.md`, `data_quality_report.csv`, `pipeline.log` |

---

## Testing

```bash
# Run all tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### GitHub Actions (suggested)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r kalimati_price_forecasting/requirements.txt
      - run: pytest kalimati_price_forecasting/tests/ -v
```

---

## Results Summary

Results are generated automatically in `outputs/reports/model_comparison.md` after running the pipeline. Models are ranked by RMSE per commodity and horizon.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{malla2025kalimati,
  author  = {Malla, Sahaj Raj},
  title   = {Multi-Model Forecasting of Daily Vegetable Prices at Nepal's Kalimati Market},
  year    = {2025},
  url     = {https://github.com/sahajrajmalla/nepal-vegetable-price}
}
```

---

## References

1. Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
2. Zhang, G.P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
3. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proc. KDD*.
4. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
5. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis*. Wiley.
6. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and ML forecasting methods: Concerns and ways forward. *PLoS ONE*, 13(3).
7. Shrestha, S. & Poudel, S. (2022). Price prediction of vegetables in Nepal using machine learning. *Journal of Agriculture and Food Research*.
8. Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Proc. NeurIPS*.
9. Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3).
10. Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study. *Expert Systems with Applications*.

---

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](../LICENSE) file for details.
