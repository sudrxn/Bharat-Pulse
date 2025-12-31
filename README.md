# 🇮🇳 Bharat-Pulse: AI-Powered Agricultural Price Predictor

**Bharat-Pulse** is an autonomous predictive engine designed to tackle food inflation in India. It forecasts the 7-day future "Modal Price" for essential commodities like Tomato, Onion, and Potato (TOP) across 100+ Indian districts.

🔗 **Live Dashboard:** (https://bharat-pulse-zdwv7vzarjgyweoky2jr9l.streamlit.app/)

## 🚀 The Problem
Agricultural markets in India are highly volatile. Sudden price spikes in "TOP" crops impact millions of households. This project provides an early-warning system for procurement officers and consumers to anticipate market shifts.

## 📊 Model Performance & Metrics
The model was trained on 600,000+ historical records (2023-2025) from Agmarknet.
- **R-Squared (R²):** 0.68 (Captures 68% of price variance)
- **Mean Absolute Error (MAE):** ₹561 (Acceptable margin for high-value commodities)
- **Trend Recall (Price Up):** 74% (High reliability in catching inflation spikes)

## 🧠 Tech Stack & Methodology
- **Predictive Engine:** Random Forest Regressor (Scikit-Learn).
- **Hyper-Tuning:** Optimized using `RandomizedSearchCV` (Best depth: None, split: 10).
- **Data Pipeline:** Custom handling for mixed date formats, state-district label encoding, and 7-day lag feature engineering.
- **Frontend:** Streamlit Cloud for real-time risk assessment.

## 📂 Project Structure
- `app.py`: The interactive dashboard.
- `train_model.py`: Engine for model building and serialization.
- `evaluate_model.py`: Script to generate R², MAE, and Confusion Matrix.
- `models/`: Contains the trained `.pkl` brains and evaluation visuals.
- `requirements.txt`: Environment dependencies.

## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/sudrxn/Bharat-Pulse.git`
2. Install libraries: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`

---
*Developed by sudrxn as a data science portfolio project.*
