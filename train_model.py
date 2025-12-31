import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_engine():
    print("🧠 Bharat-Pulse: Loading Cleaned Data...")
    
    file_path = 'processed/cleaned_master_data.csv'
    if not os.path.exists(file_path):
        print("❌ Error: Run preprocess.py first!")
        return
    
    # Memory Tip: Only load columns we need to save RAM
    cols_to_use = ['STATE', 'District Name', 'Commodity', 'Modal_Price', 'price_lag_7', 'month', 'target_7d']
    df = pd.read_csv(file_path, usecols=cols_to_use)

    print("🏷️  Encoding labels...")
    le_state = LabelEncoder()
    le_district = LabelEncoder()
    le_commodity = LabelEncoder()

    df['state_enc'] = le_state.fit_transform(df['STATE'])
    df['district_enc'] = le_district.fit_transform(df['District Name'])
    df['commodity_enc'] = le_commodity.fit_transform(df['Commodity'])

    # Define Features and Target
    features = ['state_enc', 'district_enc', 'commodity_enc', 'Modal_Price', 'price_lag_7', 'month']
    X = df[features]
    y = df['target_7d']

    # 🏗️ MEMORY OPTIMIZED MODEL SETTINGS
    print("🏗️  Building Memory-Efficient Model...")
    model = RandomForestRegressor(
        n_estimators=50,           # Reduced from 100 to 50
        max_depth=12,              # Limits how 'tall' the trees grow (Saves massive RAM)
        min_samples_leaf=5,        # Prevents over-complex branches
        random_state=42, 
        n_jobs=1                   # Use 1 core to prevent memory duplication on Windows
    )

    print("🚀 Training starting (should be much lighter now)...")
    model.fit(X, y)

    # Save to models/ folder
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(model, 'models/price_model.pkl')
    joblib.dump(le_state, 'models/le_state.pkl')
    joblib.dump(le_district, 'models/le_district.pkl')
    joblib.dump(le_commodity, 'models/le_commodity.pkl')

    print("✨ Success! Model 'Brain' is ready in the 'models/' folder.")

if __name__ == "__main__":
    train_engine()