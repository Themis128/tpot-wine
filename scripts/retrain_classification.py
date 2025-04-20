import os
import sys
import argparse
import pandas as pd
from tpot import TPOTClassifier
from dask_config import setup_local_dask

# Add project root to sys.path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    preprocess_data, split_and_resample, save_model,
    cleanup_old_models, cleanup_symlinks,
    update_leaderboards, rebuild_latest_symlinks
)
from log_utils import setup_logger

# Logger setup
logger = setup_logger()

def retrain_classification(region):
    logger.info(f"[{region}] ▶ Retraining classification model...")

    # Start Dask cluster
    client = setup_local_dask(n_workers=2, threads_per_worker=2)
    logger.info("✅   Dask client started")

    dataset_path = f"data/processed_filled/combined_{region}_filled.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"❌      Dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    X, y = preprocess_data(df)
    X_train, y_train, X_test, y_test = split_and_resample(X, y)

    model = TPOTClassifier(
        population_size=20,
        generations=5,
        max_time_mins=30,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logger.info(f"[{region}] Accuracy: {round(score, 4)}")

    model_path = save_model(model.fitted_pipeline_, "models", region)
    logger.info(f"[{region}] ✅   Model saved: {model_path}")

    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    args = parser.parse_args()

    retrain_classification(args.region)
    cleanup_old_models()
    cleanup_symlinks()
    update_leaderboards()
    rebuild_latest_symlinks()
