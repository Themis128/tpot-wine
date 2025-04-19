#!/usr/bin/env python3
import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from tpot import TPOTClassifier, TPOTRegressor
from scripts.utils.preprocessing import preprocess_data, split_and_resample
from scripts.utils.io import save_model

# === Configuration ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_filled")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LEADERBOARD_CSV = os.path.join(LOG_DIR, "leaderboard.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Setup Logging ===
log_path = os.path.join(LOG_DIR, f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)

def get_region_files():
    return [f for f in os.listdir(DATA_DIR) if f.startswith("combined_") and f.endswith("_filled.csv")]

def retrain_region(region, task="classification"):
    file_path = os.path.join(DATA_DIR, f"combined_{region}_filled.csv")
    logging.info(f"[{region}] ▶ Retraining model ({task})...")
    try:
        df = pd.read_csv(file_path)
        X, y = preprocess_data(df)

        if len(set(y)) < 2:
            raise ValueError("Target variable must have at least two classes/values.")

        X_train, y_train, X_test, y_test = split_and_resample(X, y)

        ModelClass = TPOTClassifier if task == "classification" else TPOTRegressor
        model = ModelClass(
            generations=5, population_size=20, max_time_mins=5,
            cv=3, random_state=42, n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {"region": region}
        if task == "regression":
            metrics["r2"] = round(r2_score(y_test, y_pred), 4)
            metrics["rmse"] = round(mean_squared_error(y_test, y_pred, squared=False), 4)
            logging.info(f"[{region}] R²: {metrics['r2']} | RMSE: {metrics['rmse']}")
        else:
            metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            logging.info(f"[{region}] Accuracy: {metrics['accuracy']}")

        model_path = save_model(model.fitted_pipeline_, MODEL_DIR, region)
        metrics["model_path"] = model_path
        return metrics

    except Exception as e:
        logging.error(f"[{region}] ❌ {e}")
        return {"region": region, "error": str(e)}

def main(task="classification"):
    region_files = get_region_files()
    regions = [f.replace("combined_", "").replace("_filled.csv", "") for f in region_files]

    logging.info(f"Found {len(regions)} regions for retraining.")

    results = [retrain_region(region, task) for region in regions]
    valid = [r for r in results if "model_path" in r]

    if valid:
        df = pd.DataFrame(valid)
        df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        df = df.sort_values(by="accuracy" if task == "classification" else "r2", ascending=False)
        df.to_csv(LEADERBOARD_CSV, index=False)
        logging.info(f"🏆 Leaderboard saved to {LEADERBOARD_CSV}")
    else:
        logging.warning("⚠️ No valid models were trained.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    args = parser.parse_args()
    main(task=args.task)
