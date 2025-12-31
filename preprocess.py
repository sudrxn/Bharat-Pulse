import pandas as pd
import numpy as np
import os

def clean_and_prepare():
    print("🚀 Bharat-Pulse: Starting Data Transformation...")
    
    # Path using forward slashes to avoid \A errors
    file_path = 'data/Agriculture_price_dataset.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found. Run get_data.py first!")
        return

    # Loading data
    df = pd.read_csv(file_path)
    print(f"📊 Dataset loaded: {len(df):,} rows.")
    
    # Filter for TOP crops
    df = df[df['Commodity'].isin(['Tomato', 'Onion', 'Potato'])].copy()
    
    # Standardize Dates
    # Using format='mixed' handles the '######' visual issue from Excel
    df['Price Date'] = pd.to_datetime(df['Price Date'], format='mixed', dayfirst=True)
    
    # Remove Outliers
    def remove_outliers(group):
        m, s = group['Modal_Price'].mean(), group['Modal_Price'].std()
        if s == 0: return group
        return group[(group['Modal_Price'] > m - 3*s) & (group['Modal_Price'] < m + 3*s)]

    print("🧹 Cleaning market noise and outliers...")
    df = df.groupby(['Commodity', 'District Name']).apply(remove_outliers).reset_index(drop=True)

    # Engineering signals
    df = df.sort_values(['Commodity', 'District Name', 'Price Date'])
    df['target_7d'] = df.groupby(['Commodity', 'District Name'])['Modal_Price'].shift(-7)
    df['price_lag_7'] = df.groupby(['Commodity', 'District Name'])['Modal_Price'].shift(7)
    df['month'] = df['Price Date'].dt.month
    
    df = df.dropna()
    
    if not os.path.exists('processed'): os.makedirs('processed')
    df.to_csv('processed/cleaned_master_data.csv', index=False)
    print(f"✨ Transformation successful! Saved to 'processed/cleaned_master_data.csv'")

if __name__ == "__main__":
    clean_and_prepare()