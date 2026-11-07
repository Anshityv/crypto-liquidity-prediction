# Cryptocurrency Liquidity Prediction for Market Stability

## Project Overview
This repository contains the complete machine learning pipeline, Exploratory Data Analysis (EDA), feature engineering scripts, trained model artifacts, Streamlit web application, and design documentation for predicting cryptocurrency liquidity levels.

## Directory Structure
- `data/`: Raw and processed cryptocurrency datasets (`coin_gecko_2016_2017.csv`, `processed_crypto_data.csv`).
- `models/`: Serialized trained Gradient Boosting model (`liquidity_model.pkl`), `StandardScaler` (`scaler.pkl`), and feature list.
- `plots/`: Generated EDA visualizations (`correlation_heatmap.png`, `liquidity_distribution.png`, `volume_vs_mktcap.png`).
- `docs/`: Project documentation (`EDA_Report.md`, `HLD_Document.md`, `LLD_Document.md`, `Pipeline_Architecture.md`, `Final_Report.md`).
- `eda_and_pipeline.py`: End-to-end data preprocessing, EDA, feature engineering, model training, and evaluation script.
- `app.py`: Local Streamlit web application for testing live liquidity predictions.
- `requirements.txt`: Python package dependencies.

## Quickstart Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training & EDA pipeline (replace `data/coin_gecko_2016_2017.csv` with your full dataset if desired):
   ```bash
   python eda_and_pipeline.py
   ```
3. Launch the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
