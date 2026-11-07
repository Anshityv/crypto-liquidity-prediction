# Final Project Report
## Cryptocurrency Liquidity Prediction for Market Stability

### 1. Executive Summary
Cryptocurrency markets experience frequent price shocks amplified by sudden drops in order-book liquidity. This project implements a complete machine learning pipeline to predict asset liquidity ratios (`24h_volume / mkt_cap`) using historical market metrics, volatility indicators, and rolling volume trends.

### 2. Methodology & Feature Engineering
- **Target Metric**: `liquidity_ratio` (`24h_volume / mkt_cap`), representing daily asset turnover.
- **Key Predictors Engineered**:
  - `volatility_index`: Captures short-to-medium term return dispersion across `1h`, `24h`, and `7d` windows.
  - `amihud_illiquidity`: Measures price impact per unit of log trading volume.
  - `ma_price_7d` & `ma_volume_7d`: Captures 7-day rolling trend momentum.
  - `log_price`, `log_volume`, `log_mkt_cap`: Compresses heavy-tailed financial distributions.

### 3. Model Evaluation Results (Test Set)
- **Selected Final Model**: Tuned `GradientBoostingRegressor`
- **Test RMSE**: `0.00503`
- **Test MAE**: `0.00290`
- **Test R² Score**: `0.95345`

### 4. Key Insights for Market Stability
1. **Early Crisis Detection**: Spikes in the Amihud illiquidity ratio paired with declining 7-day moving average volume reliably precede liquidity droughts.
2. **Risk Management Application**: Traders and exchanges can use the deployed Streamlit interface to flag tokens with predicted turnover ratios below `0.015`, adjusting position sizing and margin requirements before executing large orders.
