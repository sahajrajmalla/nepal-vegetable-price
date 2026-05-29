# Methodology

## Data Description

### Data Source
The study utilises daily wholesale price records from the Kalimati Fruits & Vegetables Market, the largest wholesale market in Kathmandu, Nepal. Two raw CSV datasets span the period from June 2013 to September 2023. 

| Dataset | Period | Records |
| :--- | :--- | :--- |
| File 1 | June 2013 – May 2021 | ~191,000 |
| File 2 | May 2021 – September 2023 | ~90,000 |

Each record contains six fields: Commodity, Date, Unit, Minimum, Maximum, and Average (daily wholesale price in Nepalese Rupees). The combined dataset contains 280,939 records covering 136 unique commodities across various categories including vegetables, fruits, spices, and fish.

### Data Combination
The two raw CSV files are merged via an automated script that performs schema detection, price cleaning (stripping currency prefixes and comma separators), date parsing, and unit harmonisation. Furthermore, it classifies each commodity into hierarchical categories (e.g., Root & Tuber Vegetables -> Root Vegetables) using a keyword-matching taxonomy. Exact duplicates on the composite key `[Commodity, Date, Minimum, Maximum, Average]` are dropped, yielding the unified dataset.

## Data Preprocessing
The preprocessing pipeline is deterministic and idempotent, ensuring full reproducibility. Every transformation step is comprehensively logged for auditability.

### Schema Normalisation and Duplicate Removal
Price columns are cleaned to handle plain numerics and `NaN` values, whilst string-based features (units and commodity names) are standardised. Exact duplicates on `[Commodity, Date]` are dropped, retaining the last occurrence to correctly resolve overlapping recording periods.

### Outlier Detection and Treatment
Outliers are identified using a per-commodity Interquartile Range (IQR) method combined with strict domain constraints. Critically, to prevent data leakage, the IQR bounds ($Q_1$ and $Q_3$) are computed **strictly on the training set** (dates $\le$ June 30, 2022). These frozen bounds are then applied to the entire dataset:

$$ \text{lower} = \max(Q_1 - k \cdot \text{IQR},\ P_{\min}), \quad \text{upper} = \min(Q_3 + k \cdot \text{IQR},\ P_{\max}) $$

where $k = 2.5$ serves as a conservative multiplier for price data, with hard bounds $P_{\min} = 1.0$ and $P_{\max} = 1000.0$ Rs/kg. Flagged values are winsorised (capped) rather than removed, preserving temporal continuity across the time series. Across all commodities, 3,302 values were flagged and winsorised.

### Missing Date Imputation
Each commodity is reindexed to a continuous daily date range from its first to its last recorded date. Missing values are imputed using a two-stage strategy: (1) forward-fill with a maximum gap limit of 7 days, and (2) linear interpolation. To absolutely guarantee zero target leakage, this interpolation is performed **independently within the training and testing partitions**. No interpolation crosses the train/test split boundary. In total, 121,599 daily records were imputed.

### Kalimati Vegetable Price Index (KVPI) Construction
Rather than forecasting individual commodity prices independently, this study constructs a composite daily price index—the Kalimati Vegetable Price Index (KVPI). This methodological choice smooths individual commodity fluctuations and provides a robust, singular benchmark time series.

Commodities with $\ge 365$ days of available data are retained (resulting in 135 qualifying commodities). For each qualifying commodity $c$, a base mean $\bar{P}_c^{\text{base}}$ is computed from its first 30 days of data. The normalised price is computed as:

$$ P_{c,t}^{\text{norm}} = \frac{P_{c,t}}{\bar{P}_c^{\text{base}}} \times 100 $$

Inverse-volatility weighting is applied such that commodities with a high coefficient of variation (CV) receive lower weight, stabilising the overall index:

$$ w_c = \frac{1 / \text{CV}_c}{\sum_{j} 1 / \text{CV}_j}, \quad \text{CV}_c = \frac{\sigma_c^{\text{base}}}{\bar{P}_c^{\text{base}}} $$

For each date $t$, the KVPI is the weighted average of the normalised prices across all contributing commodities $C_t$:

$$ \text{KVPI}_t = \frac{\sum_{c \in C_t} w_c \cdot P_{c,t}^{\text{norm}}}{\sum_{c \in C_t} w_c} $$

The resulting KVPI spans 3,757 days from June 2013 to September 2023, possessing a range of [91.0, 230.1].

![KVPI Price Series](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/01_kvpi_price_series.png)

## Feature Engineering
A comprehensive feature matrix of 67 predictors is constructed. All lag-based and rolling features utilise a backward shift of 1 day to strictly prevent data leakage and look-ahead bias.

*   **Autoregressive Lags (7):** Target variable lagged at 1, 2, 3, 7, 14, 21, and 30 days.
*   **Rolling Window Statistics (15):** Mean, standard deviation, minimum, maximum, and median computed over trailing windows of size $w \in \{7, 14, 30\}$ days.
*   **Exponentially Weighted Moving Averages (3):** EWMA with span parameters $\alpha \in \{7, 14, 30\}$.
*   **Differencing (2):** First-order (daily) and seasonal (weekly) differences.
*   **Calendar Features (12):** Integer encodings for day of the week, month, day of the year, quarter, and week of the year. Because Nepal observes Saturday as the primary weekly holiday, a custom `is_weekend` binary flag is implemented. Cyclical sine and cosine transformations are applied to the day of the year, day of the week, and month.
*   **Nepal Festival Dummy Variables (8):** Binary indicators for six major Nepali festivals (Dashain, Tihar, Chhath, Holi, Teej, and Nepali New Year). These incorporate specific lead and lag windows (e.g., 7 days lead and 3 days lag for Dashain) to capture pre- and post-festival price effects. A composite `fest_any` flag and a weekend interaction term are also included.
*   **Price-Derived Features (7):** Metrics such as spread, price velocity, 7-day price momentum, acceleration, squared returns (serving as a volatility clustering proxy), and rolling CVs for 7, 14, and 30 days.
*   **STL Decomposition (3):** Seasonal and Trend decomposition using Loess (STL). *Note: This feature is disabled by default in the final pipeline. Because STL applies a bidirectional global smoother across the time series, applying it prior to forecasting leaks future test-set target variance backward into the training features.*

![KVPI STL Decomposition](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/02_kvpi_decomposition.png)
![KVPI Festival Heatmap](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/03_kvpi_festival_heatmap.png)

## Experimental Setup

### Train/Test Split and Evaluation Horizons
A fixed date boundary is utilised to partition the data, yielding a training period from June 16, 2013, to June 30, 2022 (3,302 days), and a test period from July 1, 2022, to September 28, 2023 (455 days)—an approximate 88/12 percentage split. Models are evaluated at four discrete forecasting horizons: $h \in \{7, 14, 30, 90\}$ days.

### Forecasting Strategy and Metrics
For evaluating out-of-sample performance, the pipeline implements a rigorous **recursive multi-step forecasting strategy**. At each step $t$ in the test set, any required lag or rolling features are recalculated using the model's *own past predictions* rather than actual future prices. Before the recursive loop begins, all test-set targets are masked with `NaN` to guarantee zero look-ahead bias. 

The evaluation metrics computed include Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), symmetric MAPE (sMAPE), and the Coefficient of Determination ($R^2$). 

### Reproducibility and Computational Environment
Full reproducibility is ensured via a global random seed (42) propagated across Python, NumPy, PyTorch, and Optuna. PyTorch computations explicitly enforce determinism. All experiments were executed in a Python 3.13 environment running on a macOS ARM64 processor.

## Models

### Baseline and Statistical Models
The baseline models include a Naïve (random-walk) forecast and a Seasonal Naïve forecast (weekly cycle, $m=7$). The statistical models encompass Auto-ARIMA, which identified an ARIMA(2,1,1)×(0,0,0,7) structure via step-wise AIC minimisation, and a manually fitted SARIMA leveraging identical orders via maximum likelihood estimation.

![KVPI ACF PACF](/Users/sahajrajmalla/Documents/nepal-vegetable-price/kalimati_price_forecasting/outputs/report_figures/04_kvpi_acf_pacf.png)

### Machine Learning Models
The machine learning models (Random Forest, Extra Trees, Histogram-based Gradient Boosting, and XGBoost) utilise the complete 67-feature set. Hyperparameters were rigorously optimised using Optuna (Tree-structured Parzen Estimator) over 100 trials employing 5-fold TimeSeriesSplit cross-validation. For instance, the optimal XGBoost parameters included 1000 estimators, a maximum depth of 3, and a learning rate of 0.0122.

### Deep Learning Models
To ensure a mathematically fair comparison, LSTM and GRU architectures were provided the **full 67-feature matrix** (identical to the ML models). Hyperparameters (including number of layers, hidden units per layer, dropout, learning rate, and batch size) were rigorously optimized using Optuna (Tree-structured Parzen Estimator) over **100 trials**, exactly matching the tuning budget allocated to the tree-based models. Sequences of length 30 were constructed, and training employed the Adam optimizer (MSE loss), ReduceLROnPlateau scheduling, and early stopping.

### Hybrid Models
Two hybrid architectures based on the Zhang (2003) decomposition framework were evaluated:
1.  **ARIMA-LSTM:** Utilises ARIMA for linear autocorrelation extraction, followed by a ResidualLSTM architecture to model the remaining nonlinear residual structure.
2.  **ARIMA-HistGB:** Augments the ML feature matrix with ARIMA's in-sample fitted values, which are then trained via a HistGB model to capture linear and non-linear patterns concurrently.

### State-of-the-Art (SOTA) Models
PatchTST (a transformer-based model) and NBEATSx (neural basis expansion with exogenous variables) were implemented using the NeuralForecast library. They were restricted to 90-step forecasts and evaluated independently from the stacking ensemble due to fixed-length prediction constraints.

### Momentum-Corrected Online Stacking Ensemble
A multi-strategy causal ensemble aggregates predictions to form the final meta-learner. Rather than relying on a static historical blend which is highly susceptible to structural breaks, we implemented a **Momentum-Corrected Online Meta-Learner**.

At time $t$, the ensemble strictly utilizes actual observations up to $t-1$ to compute dynamic bias corrections for the incoming forecast, ensuring zero data leakage. Crucially, the meta-learner computes the **rolling residual derivative** (the slope of recent forecast errors) between the base blend and the true price. If the slope is positive—indicating that the base models are progressively falling further behind the true price—the ensemble adds a momentum penalty proportional to the rate of error growth. 

This forces the meta-learner to aggressively "lean into" upward or downward price spikes, effectively resolving the lagging problem inherent in standard sequence and tree models during periods of high volatility. The final optimal configuration blended the predictions of XGBoost and GRU, combining XGBoost's high-frequency precision with GRU's underlying trend recognition, dynamically corrected by the momentum mechanism.
