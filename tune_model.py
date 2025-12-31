import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib

def tune_engine():
    print("🔧 Bharat-Pulse: Starting Hyperparameter Tuning...")
    df = pd.read_csv('processed/cleaned_master_data.csv').sample(n=50000, random_state=42) # Sample for speed
    
    # Pre-calculated encodings for tuning
    for col, pkl in [('STATE', 'state'), ('District Name', 'district'), ('Commodity', 'commodity')]:
        le = joblib.load(f'models/le_{pkl}.pkl')
        df[f'{pkl}_enc'] = le.transform(df[col])

    X = df[['state_enc', 'district_enc', 'commodity_enc', 'Modal_Price', 'price_lag_7', 'month']]
    y = df['target_7d']

    # The "Grid" of settings to try
    param_dist = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    # n_iter=5 means it tries 5 random combinations from the grid
    tuned_search = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, scoring='r2', n_jobs=-1)
    
    print("⌛ Searching for the best settings (this may take 2-3 minutes)...")
    tuned_search.fit(X, y)
    
    print(f"✅ Best Settings Found: {tuned_search.best_params_}")
    print(f"📈 Best Tuning Score: {tuned_search.best_score_:.4f}")
    
    # Save the absolute best one
    joblib.dump(tuned_search.best_estimator_, 'models/price_model.pkl')
    print("✨ Model updated with fine-tuned parameters!")

if __name__ == "__main__":
    tune_engine()