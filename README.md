# 🚗 Used Car Price Estimation

## 📌 Project Overview

This project builds a used car price estimation system on a large (~460k rows) marketplace dataset.

The goal was not only accurate prediction, but also realistic behavior in a noisy real-world setting.

---

## 🧠 Key Challenges

- Highly skewed price distribution
- Noisy marketplace data
- Missing values
- Inconsistent model naming
- High-cardinality categorical features

---

## 🛠 Data Processing

- Removed invalid prices (0, unrealistic values)
- Filtered extreme mileage and year values
- Created engineered features:
  - `age`
  - `log_odometer`
- Normalized model names using an external validator source
- Built `base_model` column to reduce noise in model categories

---

## 🤖 Modeling

Main model:
- **HistGradientBoostingRegressor**

Target:
- `log_price = log(1 + price)`

Additional:
- Quantile regression (10% / 90%) for price interval prediction

Evaluation metrics:
- RMSE (log scale)
- MAE (log scale)
- MAE in dollars

---

## 📊 Results

The final model produces:

- Point estimate (median price)
- Lower and upper price bounds

This better reflects real marketplace uncertainty.

---

## 🖥 Streamlit UI

A simple interface allows users to:

- Enter car parameters
- Receive price prediction
- View price interval

Run with:

```bash
streamlit run app/streamlit_app.py
