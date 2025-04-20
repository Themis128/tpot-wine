#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import platform
import subprocess
from datetime import datetime

# Ensure parent directory is in Python path for utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from joblib import load
from utils import preprocess_data, generate_pdf_from_dict
from log_utils import setup_logger




logger = setup_logger()

MODEL_DIR = "models/latest"
LOG_DIR = "logs"
PREDICTION_DIR = "batch_predictions"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)


def setup_logging(verbose=False):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"predict_{now}.log")
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging to: {log_file}")


def find_latest_model(model_dir, region, backend):
    base = os.path.join(model_dir, f"{region}")
    if backend == "tpot":
        joblib_path = os.path.join(model_dir, f"combined_{region}_filled.joblib")
        if os.path.exists(joblib_path):
            return joblib_path
        alt = os.path.join(model_dir, f"{region}.joblib")
        if os.path.exists(alt):
            return alt
    elif backend == "mljar":
        zip_path = os.path.join(model_dir, f"{region}.zip")
        if os.path.exists(zip_path):
            return zip_path
    elif backend == "xgboost":
        for ext in [".json", ".bst"]:
            xgb_path = os.path.join(model_dir, f"{region}{ext}")
            if os.path.exists(xgb_path):
                return xgb_path
    raise FileNotFoundError(f"No model found for region '{region}' using backend '{backend}'")


def load_model_for_prediction(path, backend):
    if backend == "tpot":
        return load(path)
    elif backend == "mljar":
        from supervised.automl import AutoML
        return AutoML(results_path=os.path.dirname(path))
    elif backend == "xgboost":
        import xgboost as xgb
        return xgb.Booster(model_file=path)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def predict_with_model(model, X, backend):
    if backend == "tpot" or backend == "mljar":
        return model.predict(X)
    elif backend == "xgboost":
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)
    else:
        raise ValueError("Unknown backend for prediction")


def open_file(path):
    try:
        if platform.system() == "Darwin":
            subprocess.call(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)
        else:
            subprocess.call(["xdg-open", path])
    except Exception as e:
        logging.warning(f"Could not open file: {e}")


def load_and_predict(region, data_path, output_dir, backend):
    logging.info(f"[{region}] ▶ Starting prediction")

    df = pd.read_csv(data_path)
    X, _ = preprocess_data(df)

    model_path = find_latest_model(MODEL_DIR, region, backend)
    model = load_model_for_prediction(model_path, backend)
    y_pred = predict_with_model(model, X, backend)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_csv = os.path.join(output_dir, f"{region}_predictions_{timestamp}.csv")
    report_pdf = os.path.join(output_dir, f"{region}_predict_report_{timestamp}.pdf")

    df_out = df.copy()
    df_out["prediction"] = y_pred
    df_out.to_csv(pred_csv, index=False)

    stats = {
        "Region": region,
        "Backend": backend,
        "Rows predicted": len(y_pred),
        "Model used": os.path.basename(model_path)
    }

    generate_pdf_from_dict(
        title=f"Prediction Report - {region}",
        data_dict=stats,
        filename=report_pdf
    )

    logging.info(f"[{region}] ✅ Prediction CSV saved: {pred_csv}")
    logging.info(f"[{region}] ✅ Prediction PDF saved: {report_pdf}")
    open_file(report_pdf)


def main():
    parser = argparse.ArgumentParser(description="Batch prediction using AutoML models")
    parser.add_argument("--region", help="Region name")
    parser.add_argument("--data", help="Path to input CSV")
    parser.add_argument("--output_dir", default=PREDICTION_DIR)
    parser.add_argument("--backend", choices=["tpot", "mljar", "xgboost"], default="tpot")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--all", action="store_true", help="Predict for all regions")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.all:
        data_dir = "data/processed_filled"
        files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "combined_" in f]
        for file in files:
            region = file.replace("combined_", "").replace("_filled.csv", "")
            path = os.path.join(data_dir, file)
            try:
                logging.info(f"\n▶ Predicting for region: {region}")
                load_and_predict(region, path, args.output_dir, args.backend)
            except Exception as e:
                logging.error(f"[{region}] ❌ Failed: {e}")
        return

    if not args.region or not args.data:
        parser.error("--region and --data are required unless --all is used")

    try:
        load_and_predict(args.region, args.data, args.output_dir, args.backend)
    except Exception as e:
        logging.error(f"❌   Error: {e}")


if __name__ == "__main__":
    main()
