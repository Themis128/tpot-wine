import os
import joblib
import pandas as pd
from datetime import datetime
from scripts.utils.utils import preprocess_data, split_and_resample, generate_combined_validation_pdf

LOG_DIR = "logs"
MODEL_DIR = "models"
DATA_DIR = "data"

os.makedirs(LOG_DIR, exist_ok=True)

def validate_model(model_path, data_path):
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        X, y = preprocess_data(df)
        _, _, X_test, y_test = split_and_resample(X, y)
        score = model.score(X_test, y_test)
        return score
    except Exception as e:
        print(f"❌ Error validating {model_path}: {e}")
        return None

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"validation_{timestamp}.log")
    print(f"Validation log: {log_file}")

    leaderboard_path = os.path.join(LOG_DIR, "leaderboard_r2.csv")
    results = []

    print("▶ Validating models...")
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".joblib"):
            continue

        model_name = fname.replace(".joblib", "")
        model_path = os.path.join(MODEL_DIR, fname)
        data_name = model_name + "_filled.csv"
        data_path = os.path.join(DATA_DIR, data_name)

        if not os.path.exists(data_path):
            print(f"⚠️  Skipping {fname} — data file missing: {data_name}")
            continue

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

    leaderboards = {k: v for k, v in leaderboards.items() if os.path.exists(v)}

    if leaderboards:
        master_pdf = os.path.join(LOG_DIR, "validation_master_report.pdf")
        generate_combined_validation_pdf(leaderboards, master_pdf)
        print(f"\n✅ Master summary saved: {master_pdf}")
    else:
        print("\n⚠️ No leaderboard data found — master summary not generated.")

if __name__ == "__main__":
    main()
