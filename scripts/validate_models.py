import os, joblib, logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, r2_score
from scripts.utils.preprocessing import preprocess_data

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "processed_filled")
MODEL_DIR = os.path.join(BASE, "models")
LOG_DIR = os.path.join(BASE, "logs")
LEADERBOARD_PATH = os.path.join(LOG_DIR, "leaderboard.csv")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")
print(f"Validation log: {log_path}")

def validate_model(model_path, dataset_path):
    try:
        region = os.path.basename(model_path).replace(".joblib", "")
        model = joblib.load(model_path)
        df = pd.read_csv(dataset_path)
        if "wine_quality_score" not in df.columns:
            raise ValueError("Missing target column.")
        X, y = preprocess_data(df)
        y_pred = model.predict(X)
        result = {
            "region": region,
            "model_path": model_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        if y.dtype.kind in "ifu":
            result["r2"] = round(r2_score(y, y_pred), 4)
            result["score_type"] = "r2"
            logging.info(f"[{region}] R²: {result['r2']}")
        else:
            result["accuracy"] = round(accuracy_score(y, y_pred), 4)
            result["score_type"] = "accuracy"
            logging.info(f"[{region}] Accuracy: {result['accuracy']}")
        return result
    except Exception as e:
        logging.error(f"[{model_path}] ❌  {e}")
        return {"region": region, "error": str(e)}

def update_leaderboard(results):
    try:
        if os.path.exists(LEADERBOARD_PATH):
            current = pd.read_csv(LEADERBOARD_PATH)
            current = current[~current["region"].isin([e["region"] for e in results])]
            updated = pd.concat([current, pd.DataFrame(results)], ignore_index=True)
        else:
            updated = pd.DataFrame(results)
        updated.to_csv(LEADERBOARD_PATH, index=False)
        print("✅ Leaderboard updated.")
    except Exception as e:
        print(f"⚠️ Failed to update leaderboard: {e}")

def main():
    print("▶ Validating models...")
    entries = []
    for model_file in os.listdir(MODEL_DIR):
        if not model_file.endswith(".joblib"): continue
        region = model_file.replace(".joblib", "")
        data_file = f"combined_{region}_filled.csv"
        model_path = os.path.join(MODEL_DIR, model_file)
        dataset_path = os.path.join(DATA_DIR, data_file)
        if not os.path.exists(dataset_path):
            logging.warning(f"[{region}] Missing dataset.")
            continue
        result = validate_model(model_path, dataset_path)
        if "error" not in result:
            entries.append(result)
    if entries:
        update_leaderboard(entries)
    else:
        print("⚠️ No valid results.")

if __name__ == "__main__":
    main()
