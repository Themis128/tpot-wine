import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- File paths ---
CLEAN_DATA_PATH = "/home/tbaltzakis/tpot-wine/AutoML_1/39_RandomForest/data/cleaned_combined_data.csv"
MODEL_OUTPUT_PATH = "models/version_latest/model.pkl"
FEATURE_OUTPUT_PATH = "models/version_latest/feature_names.txt"
TIMESTAMP_OUTPUT_PATH = "models/version_latest/model_timestamp.txt"

# --- Rename Map (keep in sync with app) ---
RENAME_MAP = {
    'temperature_2m_mean': 'Avg Temp (2m)',
    'precipitation_sum': 'Total Precipitation',
    'et0_fao_evapotranspiration_x': 'ET0 FAO (X)',
    'et0_fao_evapotranspiration_y': 'ET0 FAO (Y)',
    'wine_quality_score': 'Wine Quality',
    'region': 'Region',
    'date': 'Date'
}

TARGET = "Wine Quality"

# --- Step 1: Load dataset ---
print("�� Loading cleaned dataset...")
df = pd.read_csv(CLEAN_DATA_PATH, decimal=",")  # Handle EU-style formatting

# --- Step 2: Rename columns for readability ---
df = df.rename(columns=RENAME_MAP)

# --- Step 3: Drop rows where target is missing ---
df = df.dropna(subset=[TARGET])
print(f"🧹 Dropped rows with missing target. Remaining: {len(df)}")

# --- Step 4: Drop non-numeric, non-predictive columns ---
non_numeric = df.select_dtypes(exclude='number').columns.tolist()
if TARGET in non_numeric:
    non_numeric.remove(TARGET)
if non_numeric:
    print(f"🗂️ Dropping non-numeric columns: {non_numeric}")
df = df.drop(columns=non_numeric)

# --- Step 5: Fill remaining missing values with column means ---
df = df.fillna(df.mean(numeric_only=True))
print("🧪 Filled missing numeric values with column means.")

# --- Step 6: Prepare train/test split ---
X = df.drop(columns=[TARGET])
y = df[TARGET]
print(f"📊 Training on {len(X)} rows with {X.shape[1]} features...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 7: Train model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 8: Save model ---
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)  # Ensure the directory exists
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"💾 Model saved to {MODEL_OUTPUT_PATH}")

# --- Step 9: Save feature names ---
os.makedirs(os.path.dirname(FEATURE_OUTPUT_PATH), exist_ok=True)  # Ensure the directory exists
with open(FEATURE_OUTPUT_PATH, "w") as f:
    for col in X.columns:
        f.write(f"{col}\n")
print(f"🧠 Feature names saved to {FEATURE_OUTPUT_PATH}")

# --- Step 10: Save timestamp ---
os.makedirs(os.path.dirname(TIMESTAMP_OUTPUT_PATH), exist_ok=True)  # Ensure the directory exists
with open(TIMESTAMP_OUTPUT_PATH, "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"🕒 Timestamp saved to {TIMESTAMP_OUTPUT_PATH}")

print("✅ Model retrained and saved successfully.")
