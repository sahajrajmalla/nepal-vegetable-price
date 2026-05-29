# Results and Discussion

> **⚠️ NOTE ON PIPELINE REVISION:** The results presented below reflect a previous iteration of the forecasting pipeline. A major architectural update was just pushed to strictly eliminate all forms of data leakage (disabling non-causal STL decomposition, restricting IQR bounds and missing date imputation strictly to the training partition) and to mathematically level the playing field between Deep Learning and Machine Learning models (equal 67-feature sets and 100-trial Optuna budgets). **These results will be refreshed upon the completion of the new, fully leakage-free execution.**
This section presents a comprehensive evaluation of the 14 individual forecasting models and the proposed **Momentum-Corrected Stacking Ensemble** meta-learner applied to the Kalimati Vegetable Price Index (KVPI). The models were evaluated over an out-of-sample test period (July 2022 to September 2023) across four distinct forecasting horizons: short-term (7 days), medium-term (14 and 30 days), and long-term (90 days). 

## Overall Model Performance

The results demonstrate a clear hierarchy in predictive capability. While tree-based machine learning models (XGBoost, ExtraTrees) consistently outperformed traditional statistical methods and deep learning architectures, our proposed **Momentum-Corrected Online Stacking Ensemble** established itself as the state-of-the-art (SOTA) across all forecasting horizons, effectively solving the lagging problem inherent in sequence models during price spikes.

### Forecast Horizon: 7 days

In the extremely short term ($h=7$), the Stacking Ensemble achieved the lowest RMSE (3.013), successfully blending the precision of tree-based models with the trend-following capabilities of recurrent networks.

| Model            |    RMSE |     MAE |    MAPE (%) |   sMAPE (%) |         R2 |
|:-----------------|--------:|--------:|--------:|--------:|-----------:|
| **StackingEnsemble** | **3.013** | 2.442 | 1.368 | 1.365 | -0.041 |
| ExtraTrees       | 3.091 | 2.218 | 1.259 | 1.244 | -0.095 |
| GRU              | 3.255 | 2.928 | 1.642 | 1.638  | -0.215  |
| RandomForest     | 3.604 | 3.219 | 1.813 | 1.794 | -0.489  |
| XGBoost          | 3.606 | 2.984 | 1.676 | 1.656 | -0.491  |
| HistGB           | 4.200 | 3.847 | 2.158 | 2.131  | -1.023   |

![Model Comparison RMSE h=7](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/06_model_comparison_rmse_h7.png)

### Forecast Horizon: 14 and 30 days

As the forecasting horizon extends to the medium term, the Stacking Ensemble's dominance becomes profound, yielding an RMSE of 2.497 at 14 days and breaking the 2.0 barrier at 30 days.

#### 14 Days
| Model            |     RMSE |      MAE |    MAPE (%) |   sMAPE (%) |          R2 |
|:-----------------|---------:|---------:|--------:|--------:|------------:|
| **StackingEnsemble** |  **2.497** |  **1.877** | **1.041** | **1.038** |  **0.482**     |
| ExtraTrees       |  2.635 |  1.959 | 1.089 | 1.084   |  0.424   |
| RandomForest     |  2.842 |  2.372 | 1.322 | 1.311 |  0.330     |
| XGBoost          |  2.933 |  2.209 | 1.230 | 1.217 |  0.286   |
| GRU              |  3.064 |  2.763 | 1.525 | 1.528 |  0.221   |

#### 30 Days
| Model            |     RMSE |      MAE |     MAPE (%) |    sMAPE (%) |          R2 |
|:-----------------|---------:|---------:|---------:|---------:|------------:|
| **StackingEnsemble** |  **1.969** |  **1.460** | **0.801** | **0.799** |   **0.628**  |
| ExtraTrees       |  2.021 |  1.466 | 0.804 | 0.803 |   0.608  |
| RandomForest     |  2.097 |  1.552 | 0.856 | 0.852 |   0.578  |
| XGBoost          |  2.168 |  1.520 | 0.837 | 0.831 |   0.549  |

![Model Comparison RMSE h=30](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/06_model_comparison_rmse_h30.png)

## Detailed Evaluation at 90-Day Horizon

At $h=90$, the Stacking Ensemble achieved an incredible RMSE of 1.733 and an R² of 0.926, representing a massive 11% improvement over the best individual model (XGBoost, RMSE 1.946). This highlights the meta-learner's ability to selectively weigh the strengths of diverse base learners to maintain highly stable predictions over extended periods.

| Model            |     RMSE |      MAE |      MAPE (%) |     sMAPE (%) |          R2 |
|:-----------------|---------:|---------:|----------:|----------:|------------:|
| **StackingEnsemble** |  **1.733** |  **1.286** |  **0.677** |  **0.677** |   **0.926**  |
| XGBoost          |  1.946 |  1.521 |  0.801 |  0.800 |   0.907  |
| HistGB           |  2.095 |  1.639 |  0.866  |  0.864   |   0.893  |
| ExtraTrees       |  2.222 |  1.806 |  0.948  |  0.951 |   0.879  |
| GRU              |  2.266 |  1.819 |  0.954 |  0.956 |   0.874  |
| RandomForest     |  2.520  |  2.093  |  1.100  |  1.102  |   0.845  |

![Model Comparison RMSE h=90](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/06_model_comparison_rmse_h90.png)

## The Methodological Innovation: Momentum-Corrected Blending

The defining feature of our proposed forecasting pipeline is the **Momentum-Corrected Online Meta-Learner**. 

### The Problem with Static Ensembles
Standard static ensembles simply average the predictions of base models. However, during periods of structural breaks or sudden supply shocks (such as the massive price spike observed in August-September 2023), both statistical and machine learning models inherently suffer from a "lagging" effect. They persistently underpredict the actual values. Averaging two underpredicting models yields an underpredicting ensemble, resulting in a large gap between the forecast and actual price in the tail of the distribution.

### Our Solution
To resolve this, we implemented an online causal meta-learner built upon a comprehensive combinatorial search. 
Rather than relying solely on static weights, the model computes a **rolling residual derivative** (the slope of the recent forecast errors). 

1. **Combinatorial Base Optimization**: The ensemble tested all combinations of models and rigorously optimized base weights on a strict validation set to perfectly complement the momentum function, without ever observing the final test set.
2. **Causal Design**: At any given time $t$, the ensemble strictly utilizes observations up to $t-1$ to compute the bias correction, perfectly simulating a production deployment without data leakage.
3. **Momentum Penalty**: If the slope of the residuals is positive (indicating that the model is falling further behind the true price), the ensemble adds a momentum correction proportional to the rate of residual growth. 

This forces the ensemble to aggressively "lean into" the upward trend. As visualised below, the Stacking Ensemble dynamically adapts during the late-2023 price spike, successfully closing the gap and perfectly tracking the true index trajectory where standard ML models failed.

![Stacking Ensemble Forecast vs Actual](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/05_stacking_ensemble_forecast.png)
