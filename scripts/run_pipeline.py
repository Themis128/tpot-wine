#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from pathlib import Path

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
DEFAULT_TASK = "regression"

def run_command(label, command):
    print(f"\n==> {label}")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"\n===== {label} =====\n")
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)
    if result.returncode != 0:
        print(f"❌   Failed at: {label}. Check '{LOG_FILE}' for details.")
        sys.exit(1)
    else:
        print(f"✅   {label} completed.")

def auto_commit_notebooks():
    print("\n==> Committing notebooks...")
    try:
        subprocess.run("git add notebooks/*.ipynb", shell=True, check=True)
        subprocess.run("git commit -m 'Auto-update notebooks for latest pipelines'", shell=True, check=True)
        print("✅   Notebooks committed to Git.")
    except subprocess.CalledProcessError:
        print("⚠️  Git commit skipped (no changes or not a repo).")

def main():
    parser = argparse.ArgumentParser(description="Run full AutoML pipeline for wine-quality.")
    parser.add_argument("--task", default=DEFAULT_TASK, choices=["regression", "classification"],
                        help="Choose ML task type: regression or classification.")
    parser.add_argument("--full", action="store_true", help="Run full pipeline: training, notebooks, validation.")
    parser.add_argument("--train", action="store_true", help="Run model training only.")
    parser.add_argument("--notebooks", action="store_true", help="Export notebooks (if applicable).")
    parser.add_argument("--validate", action="store_true", help="Run model validation.")
    parser.add_argument("--report", action="store_true", help="Generate PDF validation summary report.")
    parser.add_argument("--predict", action="store_true", help="Run prediction using a saved model.")
    parser.add_argument("--deploy", action="store_true", help="Deploy models (TBD).")
    parser.add_argument("--model", help="Path to the saved model file for prediction.")
    parser.add_argument("--data", help="Path to input CSV file for prediction.")
    parser.add_argument("--output_dir", help="Directory to save prediction outputs (CSV + PDF).")

    args = parser.parse_args()
    base_path = Path(__file__).resolve().parent

    # Create log directory and log file if not exist
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("==== TPOT Wine Quality Pipeline Log ====\n")

    # Run steps
    if args.full or args.train:
        run_command("Training Models", f"python {base_path}/train_model_unified.py --task {args.task}")

    if args.full or args.notebooks:
        run_command("Generating Notebooks", f"python {base_path}/export_notebooks.py")
        auto_commit_notebooks()

    if args.full or args.validate:
        run_command("Validating Models", f"python {base_path}/validate_models.py")

    if args.report:
        run_command("Generating Validation PDF Report", f"python {base_path}/validation_report.py")

    if args.predict:
        if not args.model or not args.data:
            print("❌   You must provide --model and --data for prediction.")
            sys.exit(1)
        command = f"python {base_path}/predict.py --model {args.model} --data {args.data}"
        if args.output_dir:
            command += f" --output_dir {args.output_dir}"
        run_command("Running Predictions", command)

    if args.deploy:
        run_command("Deploying Models", "echo 'TODO: Add deployment logic or API call.'")

    print("\n✅   All selected pipeline steps completed successfully!")

if __name__ == "__main__":
    main()
