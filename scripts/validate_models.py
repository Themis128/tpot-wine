import sys
import os
import joblib
import pandas as pd
from datetime import datetime

# Add the parent directory to the Python path to locate utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import preprocess_data, split_and_resample, generate_combined_validation_pdf
from log_utils import setup_logger

# Directories
LOG_DIR = "logs"
MODEL_DIR = "models"
DATA_DIR = "data"

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = setup_logger()

def validate_model(model_path, data_path):
    """
    Validate a single model against its corresponding dataset.

    Args:
        model_path (str): Path to the saved model (.joblib file).
        data_path (str): Path to the dataset (.csv file).

    Returns:
        float: The model's score on the test set, or None if validation fails.
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Load and preprocess the dataset
        df = pd.read_csv(data_path)
        X, y = preprocess_data(df)
        _, _, X_test, y_test = split_and_resample(X, y)
        
        # Evaluate the model
        score = model.score(X_test, y_test)
        return score
    except Exception as e:
        print(f"❌ Error validating {model_path}: {e}")
        return None

def main():
    """
    Main function to validate all models in the MODEL_DIR against their corresponding datasets.
    """
    # Timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"validation_{timestamp}.log")
    print(f"Validation log: {log_file}")

    # Path to save the leaderboard
    leaderboard_path = os.path.join(LOG_DIR, "leaderboard_r2.csv")
    results = []

    print("▶ Validating models...")
    for fname in os.listdir(MODEL_DIR):
        # Skip non-model files
        if not fname.endswith(".joblib"):
            continue

        # Derive model and data paths
        model_name = fname.replace(".joblib", "")
        model_path = os.path.join(MODEL_DIR, fname)
        data_name = model_name + "_filled.csv"
        data_path = os.path.join(DATA_DIR, data_name)

        # Check if the corresponding dataset exists
        if not os.path.exists(data_path):
            print(f"⚠️  Skipping {fname} — data file missing: {data_name}")
            continue

        # Validate the model
        score = validate_model(model_path, data_path)
        if score is not None:
            print(f"[{model_name}] R²: {round(score, 4)}")
            results.append({
                "region": model_name,
                "model_path": model_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "r2": round(score, 4),
                "score_type": "r2"
            })

    # Save results to the leaderboard
    if results:
        pd.DataFrame(results).to_csv(leaderboard_path, index=False)
        print(f"✅   Leaderboard updated: {leaderboard_path}")
    else:
        print("⚠️  No valid results.")

    # === Master Summary Generation ===
    leaderboards = {
        "R2": leaderboard_path,
        # Add other metrics as needed
    }

    # Filter out missing leaderboards
    leaderboards = {k: v for k, v in leaderboards.items() if os.path.exists(v)}

    if leaderboards:
        master_pdf = os.path.join(LOG_DIR, "validation_master_report.pdf")
        generate_combined_validation_pdf(leaderboards, master_pdf)
        print(f"\n✅ Master summary saved: {master_pdf}")
    else:
        print("\n⚠️ No leaderboard data found — master summary not generated.")

if __name__ == "__main__":
    main()
