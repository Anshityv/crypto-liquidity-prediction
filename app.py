"""
Streamlit Local Web Application for Cryptocurrency Liquidity Prediction
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

st.title("Cryptocurrency Liquidity Prediction")
st.markdown(
    "Predict market liquidity levels (`24h Volume / Market Cap`) and detect early "
    "liquidity risk based on real-time price movements, volatility, and trading volume."
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/liquidity_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found! Please run `python eda_and_pipeline.py` first.")
    st.stop()

st.subheader("Enter Market Parameters")
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Current Price (USD)", min_value=0.0001, value=1250.0, format="%.4f")
    volume_24h = st.number_input("24h Trading Volume (USD)", min_value=1.0, value=45000000.0, format="%.2f")
    mkt_cap = st.number_input("Market Capitalization (USD)", min_value=1.0, value=950000000.0, format="%.2f")
    ma_price_7d = st.number_input("7-Day Moving Avg Price (USD)", min_value=0.0001, value=1210.0, format="%.4f")
    ma_volume_7d = st.number_input("7-Day Moving Avg Volume (USD)", min_value=1.0, value=42000000.0, format="%.2f")

with col2:
    pct_1h = st.number_input("1h Price Change (%)", value=0.45, format="%.2f")
    pct_24h = st.number_input("24h Price Change (%)", value=3.20, format="%.2f")
    pct_7d = st.number_input("7d Price Change (%)", value=8.50, format="%.2f")

if st.button("Predict Market Liquidity", use_container_width=True):
    log_price = np.log1p(price)
    log_volume = np.log1p(volume_24h)
    log_mkt_cap = np.log1p(mkt_cap)
    volatility_index = float(np.std([pct_1h, pct_24h, pct_7d]))
    amihud_illiquidity = abs(pct_24h) / (log_volume + 1e-8)

    input_df = pd.DataFrame([[
        log_price, pct_1h, pct_24h, pct_7d, log_volume, log_mkt_cap,
        volatility_index, amihud_illiquidity, ma_price_7d, ma_volume_7d
    ]], columns=[
        'log_price', '1h', '24h', '7d', 'log_volume', 'log_mkt_cap',
        'volatility_index', 'amihud_illiquidity', 'ma_price_7d', 'ma_volume_7d'
    ])

    scaled_input = scaler.transform(input_df)
    prediction = float(model.predict(scaled_input)[0])
    prediction = max(0.0, prediction)

    st.divider()
    st.metric(label="Predicted Liquidity Ratio (Turnover)", value=f"{prediction:.4f}")

    if prediction >= 0.05:
        st.success("High Liquidity (Stable Market): Assets can be traded easily with minimal slippage.")
    elif prediction >= 0.015:
        st.warning("Moderate Liquidity: Normal market conditions; monitor large block orders for slippage.")
    else:
        st.error("Low Liquidity Crisis Alert: High risk of sharp price slippage and market instability!")
