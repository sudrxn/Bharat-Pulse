import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
from data.state_district_map import get_state_district_map
STATE_DISTRICT_MAP = get_state_district_map("data/Agriculture_price_dataset.csv")

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Bharat-Pulse AI", page_icon="🇮🇳", layout="wide")

# Custom CSS to make it look professional
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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
    
    # In a real app,  filter districts by state. For simplicity:
    state = st.selectbox(
        "Select State",
        sorted(STATE_DISTRICT_MAP.keys())
    )

    districts_for_state = STATE_DISTRICT_MAP[state]

    district = st.selectbox(
        "Select District",
        districts_for_state
    )
    commodity = st.selectbox("Select Commodity", sorted(le_commodity.classes_))
    st.divider()
    
    st.subheader("💰 Market Intelligence")
    current_price = st.number_input("Current Modal Price (per Quintal)", min_value=100, max_value=20000, value=2000)
    last_week_price = st.number_input("Price 7 Days Ago", min_value=100, max_value=20000, value=1900)

with col2:
    st.subheader("🔮 AI PREDICTION RESULTS")
    st.caption(
        "⚠️ Forecast assumes stable conditions based on recent market trends. "
        "Sudden shocks (weather, supply, policy) may cause actual prices to differ."
    )   

    
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
        future_prices = []
        prev_price = current_price
        prev_lag = last_week_price
        
        DAILY_CAP = 0.06  # 6% max up or down per day
        for day in range(1, 8):
            future_input = pd.DataFrame([{
                'state_enc': le_state.transform([state])[0],
                'district_enc': le_district.transform([district])[0],
                'commodity_enc': le_commodity.transform([commodity])[0],
                'Modal_Price': prev_price,
                'price_lag_7': prev_lag,
                'month': datetime.now().month
            }])

            raw_prediction = model.predict(future_input)[0]

            # Apply daily growth cap (both directions)
            max_allowed = prev_price * (1 + DAILY_CAP)
            min_allowed = prev_price * (1 - DAILY_CAP)

            next_price = min(max(raw_prediction, min_allowed), max_allowed)

            future_prices.append(next_price)

            # shift prices for next day
            prev_lag = prev_price
            prev_price = next_price
        # Confidence range using Random Forest trees
        all_tree_predictions = np.array([
            tree.predict(input_data)[0] for tree in model.estimators_
        ])

        lower_bound = np.percentile(all_tree_predictions, 10)
        upper_bound = np.percentile(all_tree_predictions, 90)

        change_pct = ((prediction - current_price) / current_price) * 100
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"₹{current_price}")
        m2.metric("AI Forecast (7-Day)", f"₹{int(prediction)}", f"{change_pct:.2f}%")
        st.info(f"📊 Expected Price Range: ₹{int(lower_bound)} – ₹{int(upper_bound)}")

        # Risk Analysis Logic
        if change_pct > 15:
            st.error(f"🚨 **HIGH RISK ALERT:** Significant price hike of {change_pct:.1f}% expected in {district}. Procurement recommended immediately.")
            risk_val = "High Risk"
        elif change_pct < -10:
            st.success(f"✅ **PRICE CORRECTION:** Prices expected to drop. Good time for consumers to wait.")
            risk_val = "Low Risk"
        else:
            st.warning("⚖️ **MARKET STABLE:** No major fluctuations expected in the next week.")
            risk_val = "Stable"
            
        # Visualization
        st.write("#### Expected Price Trajectory")
        forecast_days = ['Today'] + [f'Day +{i} (from today)' for i in range(1, 8)]

        forecast_prices = [current_price] + future_prices

        chart_df = pd.DataFrame({
            'Timeline': forecast_days,
            'Price (INR)': forecast_prices
        })

        st.line_chart(chart_df.set_index('Timeline'))

# --- 4. FOOTER ---
st.divider()
st.caption("Data Source: Agmarknet (Mandi Prices) | Model: Random Forest Regressor | Built for Bharat-Pulse")