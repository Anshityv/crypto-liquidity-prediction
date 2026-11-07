# Pipeline Architecture Document
## End-to-End Data Flow from Preprocessing to Prediction

### 1. Pipeline Flow Diagram
```text
[Raw Cryptocurrency Dataset (CSV)]
               │
               ▼
┌──────────────────────────────────────────┐
│ Step 1: Data Cleaning & Imputation       │
│ - Chronological sorting by [coin, date]  │
│ - Forward/Backward fill + Median fill    │
│ - Non-positive record removal            │
└──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Step 2: Domain Feature Engineering       │
│ - Target: Liquidity Ratio (Vol / MktCap) │
│ - Volatility Index (Std of 1h, 24h, 7d)  │
│ - Amihud Illiquidity Ratio               │
│ - 7-Day Rolling Price & Volume MAs       │
│ - Logarithmic Scaling (Price, Vol, Cap)  │
└──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Step 3: Feature Normalization & Split    │
│ - 80% Train / 20% Test Split             │
│ - StandardScaler fit on Train set        │
└──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Step 4: Model Training & Hyperparameter  │
│ - Baseline comparison (Ridge, RF, GBR)   │
│ - RandomizedSearchCV on GradientBoosting │
│ - Artifact export (model.pkl, scaler.pkl)│
└──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Step 5: Streamlit Local Inference UI     │
│ - Real-time feature calculation          │
│ - Scaled prediction & Stability Alert    │
└──────────────────────────────────────────┘
```
