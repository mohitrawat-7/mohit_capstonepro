# UPES MBA Business Analytics Capstone Project in Collaboration with KPMG

# Energy Load Forecasting — TATA Power

An end-to-end machine learning pipeline to forecast residential electricity demand (MW) for TATA Power. The project covers EDA, feature engineering, model training, evaluation, and business recommendations — built on smart meter data capturing solar generation, EV charging, temperature, and voltage patterns.

---

## Problem Statement

TATA Power faced critical operational challenges:

- No unified, data-driven mechanism to forecast demand variations across residential consumers
- Increasing solar penetration and EV adoption required smarter forecasting to handle load shifts
- Voltage fluctuations during peak periods needed diagnostic visibility
- Renewable integration (solar + wind) created grid dependency dips that conventional methods couldn't anticipate

---

## Dataset Overview

The dataset is a timestamp-level smart meter dataset with the following key features:

| Feature | Description |
|---|---|
| `total_load_mw` | Target variable — total residential load in MW |
| `temperature_c` | Ambient temperature (strong load driver) |
| `solar_generation_mw` | Solar output (creates daytime load dips) |
| `wind_generation_mw` | Wind generation |
| `ev_charging_load` | EV load (key contributor to evening peaks) |
| `voltage_readings` | Grid voltage (used for stability diagnostics) |
| `residential_load_mw` | Residential segment load |
| `industrial_load_mw` | Industrial segment load |
| `day_type` | Weekday vs Weekend flag |
| `month`, `hour`, `weekday` | Extracted datetime features |

---

## Methodology

### 1. Data Cleaning
- Removed duplicate smart-meter readings and treated missing values in Net Load, Voltage, and Solar Irradiance
- Applied interpolation for time-series gaps; corrected EV charging spikes and solar irradiance anomalies
- Aligned all features on a common timestamp index with uniform hourly intervals
- Standardised units (kW → MW, voltage normalisation)

### 2. Exploratory Data Analysis

Key patterns uncovered:

- **Evening peak behaviour** — total load consistently spikes between 6–10 PM across all day types
- **Temperature-load correlation** — strong positive relationship; AC usage drives load surges above 35°C
- **Solar dip** — net residential load drops significantly during high irradiance periods (daytime offset)
- **EV contribution** — EV charging load is heavily skewed toward night hours, amplifying the evening peak
- **Weekday vs Weekend** — weekday loads consistently higher due to industrial and commercial activity

### 3. Feature Engineering

Derived features added to improve model performance:

- `load_factor`, `daily_load_variation` — demand behaviour indicators
- `ev_load_share`, `voltage_deviation` — operational diagnostic features
- `weekend_flag`, `hour_of_day`, `day_type_indicator` — temporal pattern features
- Lag values and rolling averages — for sequence-aware prediction

### 4. Modelling

**Train-Test Split:** 80% train / 20% test (time-based, no shuffling — preserves sequence integrity)

**Scaling:** StandardScaler applied to numerical features before model training

**Models trained:**

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression (Baseline) | 49.37 | 0.39 |
| **XGBoost Regressor** | **21.81** | **0.86** |

XGBoost was selected as the final model. It outperformed Linear Regression by **55.8% on MAE** and explained **86% of load variance** — compared to just 39% for the baseline. The performance gap confirms that electricity demand relationships are inherently non-linear (temperature thresholds, EV spikes, weather-load interactions) and require a model capable of capturing these without explicit feature engineering.

**XGBoost configuration:**
```python
XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
```

---

## Key Insights

- Temperature and hour-of-day are the strongest predictors of total load
- EV charging load is the primary driver of unexpected evening spikes
- Solar generation creates a reliable daytime dip in grid dependency — usable for supply scheduling
- Lag features and rolling averages improved model accuracy by capturing sequential demand patterns
- Voltage dips cluster around high EV-load and high-temperature events — actionable for grid reinforcement

---

## Business Recommendations

Based on model findings, the following operational actions are recommended for TATA Power:

1. **Peak Load Management** — Introduce targeted Demand Response Programs during evening peaks (6–10 PM), especially for EV-heavy residential clusters
2. **EV Charging Load Control** — Encourage off-peak charging through dynamic pricing; deploy smart EV chargers to smooth sudden load spikes
3. **Grid Stability Improvements** — Reinforce feeders in high-load zones with frequent voltage dips; implement voltage regulation devices (AVRs / capacitors)
4. **Solar Optimisation** — Use solar irradiance forecasts to reduce grid dependency during peak solar hours
5. **Advanced Forecasting Integration** — Deploy the model in real-time operations for day-ahead and hour-ahead load planning to reduce grid stress and improve supply reliability

---

## Tech Stack

```
Python
pandas, numpy
matplotlib, seaborn
scikit-learn (LinearRegression, StandardScaler, train_test_split, MAE, R²)
xgboost (XGBRegressor)
```


---

# Correlation Analysis:

![alt_text](https://github.com/mohitrawat-7/mohit_capstonepro/blob/main/output.png)

# Linear Regression Result: 

![](https://github.com/mohitrawat-7/mohit_capstonepro/blob/main/LR%20output.png)

# XGBoost Output: 

![](https://github.com/mohitrawat-7/mohit_capstonepro/blob/main/XGboost%20output.png)
