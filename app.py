import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Bharat-Pulse AI", page_icon="🇮🇳", layout="wide")

# Custom CSS to make it look professional
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD AI BRAIN ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/price_model.pkl')
    le_state = joblib.load('models/le_state.pkl')
    le_district = joblib.load('models/le_district.pkl')
    le_commodity = joblib.load('models/le_commodity.pkl')
    return model, le_state, le_district, le_commodity

try:
    model, le_state, le_district, le_commodity = load_assets()
except Exception as e:
    st.error("❌ Models not found. Please run train_model.py first!")
    st.stop()

# --- 3. THE UI LAYOUT ---
st.title("🇮🇳 Bharat-Pulse: AI Food Inflation Guard")
st.markdown("### Predicting price spikes before they hit the masses.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📍 Market Selection")

    state = st.selectbox(
        "Select State",
        sorted(le_state.classes_)
    )

    # Simple district selection (no dependency)
    district = st.selectbox(
        "Select District",
        sorted(le_district.classes_)
    )

    commodity = st.selectbox(
        "Select Commodity",
        sorted(le_commodity.classes_)
    )

    st.divider()

    st.subheader("💰 Market Intelligence")
    current_price = st.number_input(
        "Current Modal Price (per Quintal)",
        min_value=100,
        max_value=20000,
        value=2000
    )

    last_week_price = st.number_input(
        "Price 7 Days Ago",
        min_value=100,
        max_value=20000,
        value=1900
    )

with col2:
    st.subheader("🔮 AI Prediction Results")

    if st.button("Generate 7-Day Forecast"):
        # Prepare data for prediction
        input_data = pd.DataFrame([{
            'state_enc': le_state.transform([state])[0],
            'district_enc': le_district.transform([district])[0],
            'commodity_enc': le_commodity.transform([commodity])[0],
            'Modal_Price': current_price,
            'price_lag_7': last_week_price,
            'month': datetime.now().month
        }])

        # AI Prediction
        prediction = model.predict(input_data)[0]
        change_pct = ((prediction - current_price) / current_price) * 100

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"₹{current_price}")
        m2.metric(
            "AI Forecast (7-Day)",
            f"₹{int(prediction)}",
            f"{change_pct:.2f}%"
        )

        # Risk Analysis Logic
        if change_pct > 15:
            st.error(
                f"🚨 **HIGH RISK ALERT:** Significant price hike of "
                f"{change_pct:.1f}% expected in {district}. "
                "Procurement recommended immediately."
            )
        elif change_pct < -10:
            st.success(
                "✅ **PRICE CORRECTION:** Prices expected to drop. "
                "Good time for consumers to wait."
            )
        else:
            st.warning(
                "⚖️ **MARKET STABLE:** No major fluctuations expected in the next week."
            )

        # Visualization
        st.write("#### Expected Price Trajectory")
        chart_df = pd.DataFrame({
            'Timeline': ['Today', 'Next Week (AI Forecast)'],
            'Price (INR)': [current_price, prediction]
        })
        st.line_chart(chart_df.set_index('Timeline'))

# --- 4. FOOTER ---
st.divider()
st.caption(
    "Data Source: Agmarknet (Mandi Prices) | "
    "Model: Random Forest Regressor | "
    "Built for Bharat-Pulse"
)
