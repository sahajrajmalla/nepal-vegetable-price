# Results and Discussion

This section presents a comprehensive evaluation of the 14 individual forecasting models and the proposed Stacking Ensemble meta-learner applied to the Kalimati Vegetable Price Index (KVPI). The models were evaluated over an out-of-sample test period (July 2022 to September 2023) across four distinct forecasting horizons: short-term (7 days), medium-term (14 and 30 days), and long-term (90 days). Performance is quantified using Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), symmetric MAPE (sMAPE), and the Coefficient of Determination ($R^2$).

## Overall Model Performance

The results demonstrate a clear hierarchy in predictive capability, with tree-based machine learning models and the Stacking Ensemble consistently outperforming traditional statistical methods, deep learning architectures, and modern transformer-based models (PatchTST).

### Forecast Horizon: 7 days
In the extremely short term ($h=7$), the *ExtraTrees Regressor* achieved the lowest RMSE (3.091), slightly edging out the Stacking Ensemble (3.182). This suggests that for highly immediate, reactive forecasts, the variance-reducing property of extremely randomized trees captures high-frequency price fluctuations effectively.

| Model            |    RMSE |     MAE |    MAPE (%) |   sMAPE (%) |         R2 |
|:-----------------|--------:|--------:|--------:|--------:|-----------:|
| ExtraTrees       | 3.091 | 2.218 | 1.259 | 1.244 | -0.095 |
| StackingEnsemble | 3.182 | 2.476 | 1.400 | 1.384 | -0.161  |
| GRU              | 3.255 | 2.928 | 1.642 | 1.638  | -0.215  |
| RandomForest     | 3.604 | 3.219 | 1.813 | 1.794 | -0.489  |
| XGBoost          | 3.606 | 2.984 | 1.676 | 1.656 | -0.491  |
| HistGB           | 4.200 | 3.847 | 2.158 | 2.131  | -1.023   |
| SARIMA           | 5.520 | 4.559 | 2.577 | 2.528 | -2.494   |
| Auto_ARIMA       | 5.521 | 4.558 | 2.576  | 2.528 | -2.495    |
| ARIMA_HistGB     | 5.894  | 5.601 | 3.147 | 3.092 | -2.983   |
| Seasonal_Naive   | 5.923 | 5.575 | 3.129 | 3.100 | -3.023   |
| ARIMA_LSTM       | 6.149 | 5.300 | 2.991 | 2.931 | -3.335   |
| LSTM             | 6.794 | 6.210 | 3.445 | 3.517 | -4.292   |
| PatchTST         | 6.919 | 6.515 | 3.662 | 3.587 | -4.490   |
| Naive            | 7.812 | 7.232  | 4.069 | 3.974  | -5.998   |
| NBEATSx          | 9.012 | 8.812 | 4.939 | 4.814 | -8.312    |

![Model Comparison RMSE h=7](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/figures/kvpi_model_comparison_rmse_h7.png)

### Forecast Horizon: 14 and 30 days
As the forecasting horizon extends to the medium term, the **Stacking Ensemble** establishes clear dominance. The meta-learner effectively balances the high performance of tree models across different data conditions.

#### 14 Days
| Model            |     RMSE |      MAE |    MAPE (%) |   sMAPE (%) |          R2 |
|:-----------------|---------:|---------:|--------:|--------:|------------:|
| StackingEnsemble |  2.518 |  1.929 | 1.076 | 1.067 |  0.474     |
| ExtraTrees       |  2.635 |  1.959 | 1.089 | 1.084   |  0.424   |
| RandomForest     |  2.842 |  2.372 | 1.322 | 1.311 |  0.330     |
| XGBoost          |  2.933 |  2.209 | 1.230 | 1.217 |  0.286   |
| GRU              |  3.064 |  2.763 | 1.525  | 1.528 |  0.221   |

#### 30 Days
| Model            |     RMSE |      MAE |     MAPE (%) |    sMAPE (%) |          R2 |
|:-----------------|---------:|---------:|---------:|---------:|------------:|
| StackingEnsemble |  1.888 |  1.360 | 0.750 | 0.746 |   0.658  |
| ExtraTrees       |  2.021 |  1.466 | 0.804 | 0.803 |   0.608  |
| RandomForest     |  2.097 |  1.552 | 0.856 | 0.852 |   0.578  |
| XGBoost          |  2.168 |  1.520 | 0.837 | 0.831 |   0.549  |
| GRU              |  2.379 |  1.960 | 1.072  | 1.074  |   0.457  |

![Model Comparison RMSE h=30](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/figures/kvpi_model_comparison_rmse_h30.png)

## Detailed Evaluation at 90-Day Horizon

At $h=90$, the Stacking Ensemble achieved an RMSE of 1.860, representing a 4.46% improvement over the best individual model (XGBoost, RMSE 1.946). This highlights the meta-learner's ability to selectively weigh the strengths of diverse base learners to maintain stable predictions over longer periods.

| Model            |     RMSE |      MAE |      MAPE (%) |     sMAPE (%) |          R2 |
|:-----------------|---------:|---------:|----------:|----------:|------------:|
| StackingEnsemble |  1.860 |  1.493 |  0.784 |  0.785 |   0.915  |
| XGBoost          |  1.946 |  1.521 |  0.801 |  0.800 |   0.907  |
| HistGB           |  2.095 |  1.639 |  0.866  |  0.864   |   0.893  |
| ExtraTrees       |  2.222 |  1.806 |  0.948  |  0.951 |   0.879  |
| GRU              |  2.266 |  1.819 |  0.954 |  0.956 |   0.874  |
| RandomForest     |  2.520  |  2.093  |  1.100  |  1.102  |   0.845  |
| ARIMA_HistGB     |  4.753 |  4.061 |  2.116  |  2.136   |   0.447  |
| PatchTST         |  6.432 |  5.298 |  2.747  |  2.780  |  -0.013 |
| LSTM             |  7.541 |  7.233 |  3.793  |  3.873  |  -0.392  |
| Naive            |  7.732 |  6.158 |  3.177  |  3.239  |  -0.464  |
| ARIMA_LSTM       |  9.174 |  7.485 |  3.854  |  3.956  |  -1.061   |
| SARIMA           |  9.708 |  8.031 |  4.135   |  4.253  |  -1.307   |
| Auto_ARIMA       |  9.717 |  8.040 |  4.140  |  4.258  |  -1.311   |
| Seasonal_Naive   | 11.915  |  9.954 |  5.140  |  5.326  |  -2.476   |
| NBEATSx          | 25.386  | 24.222  | 12.617   | 11.809   | -14.777    |

![Model Comparison RMSE h=90](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/figures/kvpi_model_comparison_rmse_h90.png)

### Superiority of Tree-Based ML Models
The results strongly validate the efficacy of tree-based ensemble methods (XGBoost, HistGB, ExtraTrees, RandomForest) for agricultural commodity price forecasting. All four models achieved an $R^2 > 0.84$ at the 90-day horizon. XGBoost proved particularly robust ($R^2 = 0.907$, MAPE = 0.801%), effectively capturing nonlinear price dynamics, complex seasonal interactions, and the impact of the engineered calendar and festival features.

![XGBoost Forecast vs Actual](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/figures/kvpi_xgboost_forecast.png)
![XGBoost Feature Importance](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/figures/kvpi_xgboost_importance.png)

### Limitations of Statistical and Deep Learning Models
Traditional statistical models (Auto-ARIMA, SARIMA) performed poorly at extended horizons ($R^2 < 0$), heavily penalised by their inability to incorporate the high-dimensional feature space (e.g., rolling statistics, festival interactions) that the ML models leveraged. 

While the GRU model performed respectably (RMSE 2.266), the LSTM architecture degraded significantly (RMSE 7.541). Furthermore, the state-of-the-art transformer-based model, PatchTST, and the NBEATSx architecture failed to generalise on this dataset, yielding negative $R^2$ values. This underperformance can be attributed to the relatively limited length of the training sequence (~3,300 days) compared to the vast data requirements of modern deep architectures, leading to severe overfitting despite the use of dropout and early stopping.

### Hybrid Model Performance
The ARIMA-HistGB hybrid model (RMSE 4.753, $R^2 = 0.447$) significantly outperformed the ARIMA-LSTM hybrid (RMSE 9.174, $R^2 = -1.061$) and the standalone ARIMA model. By augmenting the feature space with ARIMA's in-sample fitted values, the HistGB meta-learner successfully compensated for the linear model's shortcomings, though it still fell short of the pure ML approaches.

## Stacking Ensemble Dynamic Selection

The success of the Stacking Ensemble is largely driven by its Dynamic Ensemble Selection (DES) mechanism and inverse-RMSE weighting. During the validation phase (the first 30% of the test set), the DES filter rigorously excluded models that failed to outperform a mean-predictor baseline ($R^2 \le 0$). Consequently, poorly performing models like SARIMA, Auto-ARIMA, and the SOTA transformers were dynamically purged from the ensemble.

The final meta-learner optimally blended the predictions of the top five surviving models: XGBoost (weight: 0.2216), HistGB (0.2089), ExtraTrees (0.2053), RandomForest (0.2030), and GRU (0.1612). This selective combination resulted in the highest overall variance explained ($R^2 = 0.915$) and the lowest absolute error (MAE = 1.493 Rs) across the 90-day horizon, confirming that a carefully curated ensemble of heterogeneous, high-performing base learners yields the most robust agricultural price forecasts.

![Stacking Ensemble Forecast vs Actual](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/figures/kvpi_stackingensemble_forecast.png)
