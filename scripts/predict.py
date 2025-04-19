import argparse
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from scripts.utils.preprocessing import preprocess_data
from scripts.utils.utils import generate_pdf_from_dict

def setup_logging(verbose=False):
    log_level = logging.DEBUG if verbose else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/predict_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info(f"Logging to: {log_path}")

def load_and_predict(model_path, data_path, output_dir=None):
    logging.info(f"Loading model: {model_path}")
    pipeline = joblib.load(model_path)
    logging.info(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    X, _ = preprocess_data(df)
    logging.info("Running predictions...")
    predictions = pipeline.predict(X)

    output_df = pd.DataFrame({"prediction": predictions})
    if not output_dir:
        output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    region = os.path.basename(model_path).split("_")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_csv = os.path.join(output_dir, f"{region}_predictions_{timestamp}.csv")
    output_df.to_csv(output_csv, index=False)
    logging.info(f"✅  Predictions saved to: {output_csv}")

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
    parser = argparse.ArgumentParser(description="Run predictions using a saved TPOT model.")
    parser.add_argument("--model", required=True, help="Path to .joblib model file.")
    parser.add_argument("--data", required=True, help="Path to CSV data file.")
    parser.add_argument("--output_dir", help="Directory to save prediction CSV and PDF.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    load_and_predict(args.model, args.data, args.output_dir)

if __name__ == "__main__":
    main()
