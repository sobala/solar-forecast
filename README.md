# Solar PV Power Forecasting

48-hour-ahead solar generation forecasting tool for a single PV site.

## Dataset

Solar power plant in India, 34 days (May-June 2020), 22 inverters, 15-minute resolution. Resampled to 30-minute intervals. Source: [Kaggle](https://www.kaggle.com/datasets/samuelkamau/solar-data)

## Approach

- Aggregated 22 inverters to site-level mean output
- Engineered 41 features: solar position (pvlib), lagged production means, rolling statistics, clear sky index
- SHAP-driven feature pruning reduced to 5 features with no accuracy loss
- Compared XGBoost and Ridge regression with systematic hyperparameter tuning

## Results

| Model | Test RMSE (kW) | Eval RMSE (kW) | Train/Test Ratio |
|---|---|---|---|
| XGBoost (tuned, 5 features) | 114.10 | 134.51 | 1.04 |
| Ridge (tuned, 41 features) | 115.81 | 147.46 | 0.97 |

XGBoost recommended for production: more consistent generalisation, interpretable via SHAP, and fast inference (~0.001s).

## Structure

- `01_eda_and_splits.ipynb` — data exploration, cleaning, train/test/eval splits
- `02_feature_engineering.ipynb` — feature creation, solar position, lags, rolling stats
- `03_modelling.ipynb` — XGBoost, Ridge, hyperparameter tuning, SHAP analysis, evaluation

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/solar-forecast.git
cd solar-forecast
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Compute

MacBook Pro M4 Pro (2024), 24GB RAM, CPU only.