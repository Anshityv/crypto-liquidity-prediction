# Exploratory Data Analysis (EDA) Report
## Cryptocurrency Liquidity Prediction for Market Stability

### 1. Dataset Summary
- **Time Period**: 2016 - 2017 Historical Cryptocurrency Market Records
- **Core Attributes**: `coin`, `date`, `price`, `1h`, `24h`, `7d`, `24h_volume`, `mkt_cap`
- **Engineered Features**: `liquidity_ratio`, `volatility_index`, `amihud_illiquidity`, `ma_price_7d`, `ma_volume_7d`, `log_price`, `log_volume`, `log_mkt_cap`

### 2. Key Statistical Insights & Distributions
1. **Heavy Right-Skewness in Raw Financials**: Raw `price`, `24h_volume`, and `mkt_cap` span multiple orders of magnitude across large-cap and small-cap coins. Applying a `log(1 + x)` transformation normalizes their distributions and prevents extreme values from dominating regression weights.
2. **Target Variable (`liquidity_ratio`)**: Defined as `24h_volume / mkt_cap`, the turnover ratio clusters between `0.02` and `0.12` under normal trading conditions. Values below `0.015` indicate thin order books and heightened slippage risk.
3. **Correlation Analysis**:
   - `log_volume` and `log_mkt_cap` exhibit strong positive correlation (`r > 0.85`).
   - `amihud_illiquidity` negatively correlates with `liquidity_ratio`, confirming that large absolute price swings per unit of trading volume signal illiquid market states.

### 3. Generated Visualizations
- `plots/correlation_heatmap.png`: Pearson correlation matrix of raw and engineered features.
- `plots/liquidity_distribution.png`: Histogram and KDE distribution of the target liquidity ratio.
- `plots/volume_vs_mktcap.png`: Scatter plot of Log Market Cap vs. Log 24h Volume colored by liquidity level.
