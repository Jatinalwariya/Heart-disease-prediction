import sys
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# SCALE FULL DATA for final model saving
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(rf, 'Models/random_forest_model.pkl')
joblib.dump(scaler, 'Models/scaler.pkl')

print("🔷 Random Forest model and scaler saved successfully.")