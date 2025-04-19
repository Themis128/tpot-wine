import argparse
import os
import logging
import pandas as pd
import joblib
from datetime import datetime

from scripts.utils.utils import preprocess_data, generate_pdf_from_dict


def setup_logging(verbose=False):
    log_level = logging.DEBUG if verbose else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/predict_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to: {log_path}")


def find_latest_model(model_dir, region):
    pattern = f"combined_{region}_filled_"
    models = [
        f for f in os.listdir(model_dir)
        if f.startswith(pattern) and f.endswith(".joblib")
    ]
    if not models:
        raise FileNotFoundError(f"No model found for region: {region}")
    latest_model = max(models, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
    return os.path.join(model_dir, latest_model)


def load_and_predict(region, data_path, model_dir="models", output_dir=None):
    model_path = find_latest_model(model_dir, region)
    logging.info(f"Using model: {model_path}")
    pipeline = joblib.load(model_path)

    logging.info(f"Reading data: {data_path}")
    df = pd.read_csv(data_path)
    X, _ = preprocess_data(df)

    logging.info("Running predictions...")
    predictions = pipeline.predict(X)

    output_df = pd.DataFrame({"prediction": predictions})
    output_dir = output_dir or os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"{region}_predictions_{timestamp}.csv")
    output_df.to_csv(output_csv, index=False)
    logging.info(f"✅ Predictions saved to: {output_csv}")

    summary = {
        "Region": region,
        "Model": os.path.basename(model_path),
        "Samples Predicted": len(predictions),
        "Output CSV": os.path.basename(output_csv),
        "Run Timestamp": timestamp
    }

    pdf_path = os.path.join(output_dir, f"{region}_predict_report_{timestamp}.pdf")
    generate_pdf_from_dict("Wine Quality Prediction Summary", summary, filename=pdf_path)
    logging.info(f"📄 Prediction report saved to: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Run predictions using the latest model for a region.")
    parser.add_argument("--region", required=True, help="Region name to identify model and data.")
    parser.add_argument("--data", help="Path to CSV file. Defaults to processed_filled/<region>.csv")
    parser.add_argument("--output_dir", help="Directory to save outputs.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    data_path = args.data or f"data/processed_filled/combined_{args.region}_filled.csv"
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    load_and_predict(args.region, data_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
