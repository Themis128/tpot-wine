#!/usr/bin/env python3
import os
import pandas as pd
import logging
import argparse
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tpot import TPOTRegressor, TPOTClassifier
from scripts.utils.utils import preprocess_data, split_and_resample, save_model

# === Constants ===
DATA_DIR = "data/processed_filled"
OUTPUT_DIR = "models"
LOG_DIR = "logs"
MAX_TIME_MINS = 5
GENERATIONS = 5
POPULATION_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === TPOT Configuration ===
regressor_config_dict = {
    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': ['auto'],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    },
    'sklearn.linear_model.Ridge': {'alpha': [0.1, 1.0, 10.0]},
    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': [3, 5, None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
}

classifier_config_dict = {
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'max_features': ['auto'],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    },
    'sklearn.linear_model.LogisticRegression': {
        'C': [1.0],
        'solver': ['liblinear']
    }
}

# === Logging ===
def setup_logging(verbose=False):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"train_model_{now}.log")
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to file: {log_file}")

# === Training Function ===
def train_model(file_path, task):
    region = os.path.basename(file_path).replace("combined_", "").replace(".csv", "")
    logging.info(f"[{region}] ▶ Starting training ({task})")
    try:
        df = pd.read_csv(file_path)
        if "wine_quality_score" not in df.columns:
            raise ValueError("Missing target column: wine_quality_score")
        X, y = preprocess_data(df)
        X_train, y_train, X_test, y_test = split_and_resample(X, y)

        model_cls = TPOTRegressor if task == "regression" else TPOTClassifier
        config_dict = regressor_config_dict if task == "regression" else classifier_config_dict

        model = model_cls(
            generations=GENERATIONS,
            population_size=POPULATION_SIZE,
            max_time_mins=MAX_TIME_MINS,
            config_dict=config_dict,
            random_state=42,
            n_jobs=-1,
            verbosity=2
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {"region": region}
        if task == "regression":
            metrics["r2"] = round(r2_score(y_test, y_pred), 4)
            metrics["rmse"] = round(mean_squared_error(y_test, y_pred, squared=False), 4)
            logging.info(f"[{region}] R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
        else:
            metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            logging.info(f"[{region}] Accuracy: {metrics['accuracy']:.4f}")

        model_path = save_model(model.fitted_pipeline_, OUTPUT_DIR, region)
        metrics["model_path"] = model_path
        logging.info(f"[{region}] ✅ Model saved: {model_path}")
        return metrics

    except Exception as e:
        logging.error(f"[{region}] ❌ Error: {e}")
        return None

# === Entry Point ===
def main(task, verbose, leaderboard_dir):
    setup_logging(verbose)
    leaderboard_file = os.path.join(leaderboard_dir, f"leaderboard_{task}.csv")
    os.makedirs(leaderboard_dir, exist_ok=True)

    files = [f for f in os.listdir(DATA_DIR) if f.startswith("combined_") and f.endswith(".csv")]
    logging.info(f"Found {len(files)} datasets for task: {task}")
    
    results = []
    for file in files:
        full_path = os.path.join(DATA_DIR, file)
        result = train_model(full_path, task)
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(leaderboard_file, index=False)
        logging.info(f"🏆 Leaderboard saved: {leaderboard_file}")
    else:
        logging.warning("No valid models were trained.")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["regression", "classification"])
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--leaderboard_dir", default="logs", help="Directory to save leaderboard CSV")
    args = parser.parse_args()

    main(task=args.task, verbose=args.verbose, leaderboard_dir=args.leaderboard_dir)
