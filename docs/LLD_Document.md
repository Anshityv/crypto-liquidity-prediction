# Low-Level Design (LLD) Document
## Component-Level Implementation Breakdown

| Module | Function / Script | Input | Output | Logic & Implementation Details |
| :--- | :--- | :--- | :--- | :--- |
| **Data Ingestion & Cleaning** | `load_and_clean_data(filepath)` | Raw CSV file path | Cleaned `pd.DataFrame` | Standardizes column names, converts `date` to `datetime`, sorts chronologically per coin, applies `ffill`/`bfill` and median imputation, and removes non-positive price/market-cap rows. |
| **Feature Engineering** | `engineer_features(df)` | Cleaned `pd.DataFrame` | Engineered `pd.DataFrame` | Computes `liquidity_ratio = 24h_volume / mkt_cap`, `volatility_index = std(1h, 24h, 7d)`, `amihud_illiquidity = |24h| / log1p(24h_volume)`, 7-day rolling means, and `log1p` features. Clips 99.5th percentile outliers. |
| **Exploratory Data Analysis** | `perform_eda(df)` | Engineered `pd.DataFrame` | PNG Visualizations | Generates correlation heatmap, target distribution histogram, and volume-vs-market-cap scatter plot inside `plots/`. |
| **Model Training & Tuning** | `train_and_evaluate(df)` | Engineered `pd.DataFrame` | `.pkl` model artifacts & metrics | Splits data 80/20, normalizes features via `StandardScaler`, evaluates Ridge, Random Forest, and Gradient Boosting on RMSE, MAE, and R2, tunes hyperparameters via 3-fold `RandomizedSearchCV`, and saves artifacts to `models/`. |
| **Web Deployment** | `app.py` | User UI numeric inputs | Liquidity score & risk badge | Loads `scaler.pkl` and `liquidity_model.pkl`, transforms user inputs on the fly, predicts liquidity turnover ratio, and classifies market stability into High, Moderate, or Crisis states. |
