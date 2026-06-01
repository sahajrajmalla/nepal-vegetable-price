# Kalimati Vegetable Price Forecasting

A robust, multi-horizon time-series forecasting pipeline built to predict daily wholesale vegetable prices at the Kalimati Fruits & Vegetables Market in Kathmandu, Nepal (2013-2023).

This repository contains the complete codebase for data processing, feature engineering, and the training of 14 distinct model architectures—ranging from statistical baselines to deep learning hybrids and a custom stacking ensemble meta-learner.

## Overview

Forecasting agricultural commodities in emerging markets is difficult due to extreme volatility, seasonality, and festival-driven price spikes. Instead of forecasting 100+ noisy individual items, this project constructs the **Kalimati Vegetable Price Index (KVPI)**—a custom, inverse-volatility weighted index of 135 commodities.

The pipeline predicts the KVPI across four distinct horizons: **7, 14, 30, and 90 days**.

## Key Features

- **End-to-End Pipeline**: Handles everything from raw dirty CSVs (mixed date formats, currency prefixes) to final model evaluation.
- **Robust Feature Engineering**: 67 features covering autoregressive lags, rolling statistics, and exact Nepali festival offsets (Dashain, Tihar, Chhath, Holi, Teej, New Year). *Note: Non-causal global smoothers like STL were explicitly disabled to guarantee zero data leakage.*
- **Model Diversity**: 
  - *Baselines*: Naive, Seasonal Naive
  - *Statistical*: Auto-ARIMA, SARIMA
  - *Tree-based ML*: Random Forest, ExtraTrees, HistGB, XGBoost (Bayesian optimized via Optuna, 100 trials)
  - *Deep Learning*: LSTM, GRU (PyTorch — mathematically leveled with full 67 features and identical 100-trial Optuna tuning)
  - *Hybrids*: ARIMA-LSTM, ARIMA-HistGB
  - *SOTA Transformers*: PatchTST, NBEATSx (NeuralForecast)
- **Stacking Ensemble**: A dynamic ensemble selection (DES) meta-learner that filters out poor models dynamically on a validation holdout, combining the top 5 models using inverse-RMSE weighting.

## Repository Structure

```text
kalimati_price_forecasting/
├── configs/            # YAML configuration files
├── data/               # Raw, interim, and processed datasets
├── outputs/            # Logs, evaluation reports, and model checkpoints
├── src/                # Core modules (preprocessing, feature eng, evaluation)
│   └── models/         # Model architectures
├── 01_combine_data.py  # Data ingestion and cleaning
├── 02_preprocess.py    # Pipeline entry point for KVPI creation
├── 03_train_baselines.py 
├── ...                 # Individual training scripts per model family
├── 08_train_ensemble.py# Meta-learner stacking
├── 09_evaluate.py      # Final metrics generation
├── run_all.py          # Master orchestrator script
└── clear.py            # Utility to wipe outputs/caches
```

## Getting Started

### Prerequisites
- Python 3.10+ (Tested on 3.13.5)
- macOS ARM64 / Linux / Windows

### Installation
1. Clone the repo:
```bash
git clone https://github.com/sahajrajmalla/nepal-vegetable-price.git
cd nepal-vegetable-price/kalimati_price_forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

*(Note: PyTorch, XGBoost, and Optuna are required).*

### Running the Pipeline

You can run individual stages (e.g., just `python 01_combine_data.py`), but the easiest way to replicate the full research pipeline is using the orchestrator:

```bash
python run_all.py
```

This executes the pipeline sequentially. **Note:** Full execution takes roughly 12-14 hours on a modern CPU due to the rigorous Optuna TimeSeriesSplit hyperparameter searches for the ML models.

## Results

Our top-performing model is the **Momentum-Corrected Online Stacking Ensemble** meta-learner, which achieves an $R^2$ of **0.924** at the 90-day forecasting horizon, completely solving the "lagging" effect that plagues standard sequence models during volatile price spikes. It outperforms traditional statistical approaches (ARIMA), pure deep-learning models (LSTM), and state-of-the-art transformers (PatchTST) on this specific dataset volume.



## Citation

If you use this work or codebase, please cite it as:

```bibtex
@misc{malla2026kalimativegetablepriceindex,
      title={Kalimati Vegetable Price Index Forecasting with a Momentum Corrected Online Stacking Ensemble}, 
      author={Sahaj Raj Malla},
      year={2026},
      eprint={2605.30720},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.30720}, 
}
```

## License

Distributed under the GNU Affero General Public License v3 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

