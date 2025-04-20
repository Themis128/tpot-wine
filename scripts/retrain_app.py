import os
import sys
import argparse
import pandas as pd
from tpot import TPOTClassifier, TPOTRegressor

# Add project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import preprocess_data, split_and_resample, save_model
from utils import cleanup_old_models, cleanup_symlinks, update_leaderboards
from log_utils import setup_logger

# === Setup ===
os.environ["TPOT_USE_DASK"] = "off"
logger = setup_logger()

def retrain_model(region: str, task: str = "classification"):
    logger.info(f"[{region}] ▶ Retraining model ({task})")
    dataset_path = f"data/processed_filled/combined_{region}_filled.csv"
    
    if not os.path.exists(dataset_path):
        logger.error(f"❌   Dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    X, y = preprocess_data(df)
    X_train, y_train, X_test, y_test = split_and_resample(X, y)

    if task == "classification":
        model = TPOTClassifier(generations=5, population_size=20, n_jobs=1, random_state=42)
    else:
        model = TPOTRegressor(generations=5, population_size=20, n_jobs=1, random_state=42)

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logger.info(f"[{region}] Score: {round(score, 4)}")

    # Embed task type in fitted pipeline for leaderboard use
    model.fitted_pipeline_.task_type = task
    model_path = save_model(model.fitted_pipeline_, "models", region)
    logger.info(f"[{region}] ✅  Model saved to: {model_path}")

def retrain_all_regions(task: str):
    model_dir = "models/latest"
    available_models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    available_regions = sorted([
        f.replace("combined_", "").replace("_filled.joblib", "")
        for f in available_models if f.startswith("combined_")
    ])
    for region in available_regions:
        retrain_model(region, task)

# Run maintenance
cleanup_old_models()
cleanup_symlinks()
update_leaderboards()

# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain TPOT model(s).")
    parser.add_argument("--region", help="Region name (e.g. Heraklion), or 'all' for all regions")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    args = parser.parse_args()

    if args.region == "all":
        retrain_all_regions(args.task)
    elif args.region:
        retrain_model(args.region, args.task)
    else:
        logger.error("❌   You must specify --region or use --region all")
