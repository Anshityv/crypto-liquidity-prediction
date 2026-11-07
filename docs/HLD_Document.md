# High-Level Design (HLD) Document
## Cryptocurrency Liquidity Prediction System

### 1. System Overview
The Cryptocurrency Liquidity Prediction system is an end-to-end machine learning solution designed to forecast cryptocurrency liquidity levels (`24h_volume / mkt_cap`) and detect early signs of market instability.

### 2. Architectural Components
1. **Data Ingestion & Storage Layer**:
   - Reads raw historical CSV data (`data/coin_gecko_2016_2017.csv`).
   - Outputs cleaned and feature-engineered datasets (`data/processed_crypto_data.csv`).
2. **Preprocessing & Feature Engineering Pipeline**:
   - Handles missing values using time-series group forward/backward filling and median imputation.
   - Computes rolling moving averages, multi-horizon volatility, Amihud illiquidity proxy, and logarithmic transformations.
3. **Model Training & Serialization Engine**:
   - Benchmarks linear (`Ridge`) and non-linear ensemble models (`RandomForestRegressor`, `GradientBoostingRegressor`).
   - Optimizes hyperparameters via `RandomizedSearchCV` and exports `liquidity_model.pkl` and `scaler.pkl`.
4. **Local Deployment & Inference Layer (Streamlit)**:
   - Interactive web interface (`app.py`) accepting real-time market inputs and outputting liquidity predictions with stability alerts.
