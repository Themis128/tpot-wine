import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add the parent directory to the Python path to locate utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import preprocess_data, split_and_resample, save_model, infer_task_type
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

from tpot import TPOTRegressor, TPOTClassifier
from supervised.automl import AutoML
import xgboost as xgb

from log_utils import setup_logger

# ========== CONFIG ==========
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed_filled"))
OUTPUT_DIR = "models"
LOG_DIR = "logs"
MAX_TIME_MINS = 5
GENERATIONS = 5
POPULATION_SIZE = 20
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = setup_logger()

def setup_logging():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"train_all_{now}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging to: {log_file}")

# ========== TRAINERS ==========

def train_with_tpot(X_train, y_train, X_test, y_test, task):
    model_cls = TPOTRegressor if task == "regression" else TPOTClassifier
    model = model_cls(
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        max_time_mins=MAX_TIME_MINS,
        cv=5,
        random_state=42,
        n_jobs=1  # Removed 'verbosity' argument
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model.fitted_pipeline_, y_pred, "TPOT"

def train_with_mljar(X_train, y_train, X_test, y_test, task):
    automl = AutoML(mode="Compete", eval_metric="r2" if task == "regression" else "accuracy")
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    return automl, y_pred, "MLJAR"

def train_with_xgboost(X_train, y_train, X_test, y_test, task):
    model = xgb.XGBRegressor() if task == "regression" else xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred, "XGBoost"

# ========== TRAIN RUNNER ==========

def train_all_models_for_file(file_path):
    region = os.path.basename(file_path).replace("combined_", "").replace("_filled.csv", "")
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df)
    
    # Automatically infer task type
    task = infer_task_type(y)
    logging.info(f"[{region}] Task inferred as: {task}")

    X_train, y_train, X_test, y_test = split_and_resample(X, y)

    all_results = []
    
    for trainer in [train_with_tpot, train_with_mljar, train_with_xgboost]:
        label = None  # Initialize label to avoid UnboundLocalError
        try:
            model, y_pred, label = trainer(X_train, y_train, X_test, y_test, task)
            metrics = {
                "region": region,
                "algorithm": label,
                "task": task
            }
            if task == "regression":
                metrics["r2"] = round(r2_score(y_test, y_pred), 4)
                metrics["rmse"] = round(mean_squared_error(y_test, y_pred, squared=False), 4)
            else:
                metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)

            model_path = save_model(model, os.path.join(OUTPUT_DIR, label), region)
            metrics["model_path"] = model_path
            logging.info(f"[{region}] ✅ {label} model saved: {model_path}")
            all_results.append(metrics)
        except Exception as e:
            logging.error(f"[{region}] ❌ Failed with {label or 'Unknown'}: {e}")
    return all_results

# ========== MAIN ==========

def main():
    setup_logging()
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f.startswith("combined_")]
    all_metrics = []
    for f in files:
        full_path = os.path.join(DATA_DIR, f)
        result = train_all_models_for_file(full_path)
        all_metrics.extend(result)

    out_csv = os.path.join(LOG_DIR, "all_model_results.csv")
    pd.DataFrame(all_metrics).to_csv(out_csv, index=False)
    logging.info(f"🏁 All model results saved to: {out_csv}")

if __name__ == "__main__":
    main()
