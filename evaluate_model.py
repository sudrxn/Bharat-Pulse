import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate():
    print("📊 Bharat-Pulse: Starting Model Evaluation...")
    
    # 1. Load Data and Models
    if not os.path.exists('processed/cleaned_master_data.csv'):
        print("❌ Error: processed/cleaned_master_data.csv not found!")
        return
        
    df = pd.read_csv('processed/cleaned_master_data.csv')
    
    # Load the "Brain" files
    model = joblib.load('models/price_model.pkl')
    le_state = joblib.load('models/le_state.pkl')
    le_dist = joblib.load('models/le_district.pkl')
    le_comm = joblib.load('models/le_commodity.pkl')

    # 2. Prepare Data for AI
    # We encode the names into the numbers the model understands
    df['state_enc'] = le_state.transform(df['STATE'])
    df['district_enc'] = le_dist.transform(df['District Name'])
    df['commodity_enc'] = le_comm.transform(df['Commodity'])

    features = ['state_enc', 'district_enc', 'commodity_enc', 'Modal_Price', 'price_lag_7', 'month']
    X = df[features]
    y_actual = df['target_7d']

    # 3. Generate Predictions
    print("🔮 Model is analyzing historical patterns...")
    y_pred = model.predict(X)

    # 4. REGRESSION METRICS (How accurate is the price?)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    print("\n" + "="*30)
    print("📈 REGRESSION METRICS")
    print("="*30)
    print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
    print(f"R-Squared Score: {r2:.4f}")
    print("Note: R2 > 0.60 is considered strong for agri-commodities.")

    # 5. TREND METRICS (Did we get the direction right?)
    # This is how we get the Confusion Matrix and F1-Score for your resume!
    # We define a 'Trend' as 1 if price goes UP, 0 if it stays same or goes DOWN.
    actual_trend = (y_actual > df['Modal_Price']).astype(int)
    pred_trend = (y_pred > df['Modal_Price']).astype(int)

    cm = confusion_matrix(actual_trend, pred_trend)
    
    print("\n" + "="*30)
    print("🎯 TREND PREDICTION (UP/DOWN)")
    print("="*30)
    print(classification_report(actual_trend, pred_trend, target_names=['Stable/Down', 'Price Up']))

    # 6. Save Confusion Matrix Visual
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=['Predicted Down', 'Predicted Up'], 
                yticklabels=['Actual Down', 'Actual Up'])
    plt.title('Bharat-Pulse: Trend Prediction Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    print("\n✅ Confusion Matrix image saved in 'models/confusion_matrix.png'")

if __name__ == "__main__":
    evaluate()