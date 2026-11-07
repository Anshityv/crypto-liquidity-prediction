"""
Cryptocurrency Liquidity Prediction - End-to-End ML Pipeline
Covers: Data Loading, Preprocessing, EDA, Feature Engineering, 
Model Training, Hyperparameter Tuning, and Evaluation.
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads raw cryptocurrency data, standardizes column names, and handles missing values."""
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    col_map = {
        'volume_24h': '24h_volume',
        'volume': '24h_volume',
        'market_cap': 'mkt_cap',
        'marketcap': 'mkt_cap',
        'close': 'price'
    }
    df.rename(columns=col_map, inplace=True)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        sort_cols = ['coin', 'date'] if 'coin' in df.columns else ['date']
        df.sort_values(by=sort_cols, inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'coin' in df.columns:
        df[num_cols] = df.groupby('coin')[num_cols].transform(lambda x: x.ffill().bfill())
    
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    df = df[(df['price'] > 0) & (df['mkt_cap'] > 0)]
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers domain-specific cryptocurrency liquidity features."""
    print("[INFO] Engineering liquidity features...")
    data = df.copy()

    if '24h' not in data.columns:
        data['24h'] = data.groupby('coin')['price'].pct_change(1).fillna(0) * 100 if 'coin' in data.columns else data['price'].pct_change(1).fillna(0) * 100
    if '7d' not in data.columns:
        data['7d'] = data.groupby('coin')['price'].pct_change(7).fillna(0) * 100 if 'coin' in data.columns else data['price'].pct_change(7).fillna(0) * 100
    if '1h' not in data.columns:
        data['1h'] = data['24h'] / 24.0

    data['liquidity_ratio'] = data['24h_volume'] / (data['mkt_cap'] + 1e-8)
    data['volatility_index'] = np.std(data[['1h', '24h', '7d']], axis=1)
    data['amihud_illiquidity'] = np.abs(data['24h']) / (np.log1p(data['24h_volume']) + 1e-8)

    if 'coin' in data.columns:
        data['ma_price_7d'] = data.groupby('coin')['price'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        data['ma_volume_7d'] = data.groupby('coin')['24h_volume'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    else:
        data['ma_price_7d'] = data['price'].rolling(7, min_periods=1).mean()
        data['ma_volume_7d'] = data['24h_volume'].rolling(7, min_periods=1).mean()

    data['log_price'] = np.log1p(data['price'])
    data['log_volume'] = np.log1p(data['24h_volume'])
    data['log_mkt_cap'] = np.log1p(data['mkt_cap'])

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    upper_limit = data['liquidity_ratio'].quantile(0.995)
    data['liquidity_ratio'] = np.clip(data['liquidity_ratio'], 0, upper_limit)

    data.to_csv("data/processed_crypto_data.csv", index=False)
    print("[INFO] Processed data saved to 'data/processed_crypto_data.csv'")
    return data


def perform_eda(df: pd.DataFrame):
    """Generates and saves statistical summary and visualizations."""
    print("[INFO] Generating EDA visualizations...")
    feature_cols = [
        'price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
        'volatility_index', 'amihud_illiquidity', 'liquidity_ratio'
    ]
    
    plt.figure(figsize=(10, 8))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix (Cryptocurrency Liquidity)")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df['liquidity_ratio'], bins=50, kde=True, color='teal')
    plt.title("Distribution of Market Liquidity Ratio (24h Volume / Market Cap)")
    plt.xlabel("Liquidity Ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/liquidity_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df['log_mkt_cap'], y=df['log_volume'],
        hue=df['liquidity_ratio'], palette='viridis', alpha=0.7
    )
    plt.title("Log Market Cap vs. Log 24h Trading Volume")
    plt.xlabel("Log(Market Cap)")
    plt.ylabel("Log(24h Volume)")
    plt.tight_layout()
    plt.savefig("plots/volume_vs_mktcap.png", dpi=300)
    plt.close()
    print("[INFO] EDA plots saved inside 'plots/' directory.")


def train_and_evaluate(df: pd.DataFrame):
    """Splits dataset, scales features, compares baseline models, tunes GradientBoosting, and saves artifacts."""
    features = [
        'log_price', '1h', '24h', '7d', 'log_volume', 'log_mkt_cap',
        'volatility_index', 'amihud_illiquidity', 'ma_price_7d', 'ma_volume_7d'
    ]
    target = 'liquidity_ratio'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    print("\n--- Baseline Model Performance ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name:18s} | RMSE: {rmse:.5f} | MAE: {mae:.5f} | R2: {r2:.5f}")

    print("\n[INFO] Running Hyperparameter Tuning on Gradient Boosting...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=8,
        scoring='neg_root_mean_squared_error',
        cv=3,
        random_state=42,
        verbose=0
    )
    gb_search.fit(X_train_scaled, y_train)
    best_model = gb_search.best_estimator_

    final_preds = best_model.predict(X_test_scaled)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    final_mae = mean_absolute_error(y_test, final_preds)
    final_r2 = r2_score(y_test, final_preds)

    print("\n=== Tuned Model Test Performance ===")
    print(f"Best Parameters : {gb_search.best_params_}")
    print(f"Test RMSE       : {final_rmse:.5f}")
    print(f"Test MAE        : {final_mae:.5f}")
    print(f"Test R2 Score   : {final_r2:.5f}")

    joblib.dump(best_model, "models/liquidity_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(features, "models/feature_names.pkl")
    print("[INFO] Saved model and scaler to 'models/' directory.")


if __name__ == "__main__":
    DATA_PATH = "data/coin_gecko_2016_2017.csv"
    cleaned_df = load_and_clean_data(DATA_PATH)
    featured_df = engineer_features(cleaned_df)
    perform_eda(featured_df)
    train_and_evaluate(featured_df)
