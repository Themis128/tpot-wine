import os
import subprocess
import argparse
from datetime import datetime
from log_utils import setup_logger

LOG_FILE = "logs/pipeline.log"
logger = setup_logger()

def run_pipeline_backend(task=None, full=False, train=False, validate=False,
                         notebooks=False, report=False, predict=False,
                         model=None, data=None, output_dir=None,
                         region=None, log_callback=None):
    
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            logger.info(msg)

    def log_to_file(msg):
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {msg}\n")

    def run_subprocess(cmd, step_name):
        log(f"▶ {step_name}")
        log_to_file(f"===== {step_name} =====")
        log_to_file(f"Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout.strip():
                log(result.stdout.strip())
                log_to_file(result.stdout.strip())
            log(f"✅  {step_name} completed successfully.")
            log_to_file(f"✅  {step_name} completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or str(e)
            log(f"❌  Failed: {step_name}\n{error_msg}")
            log_to_file(f"❌  Failed: {step_name}\n{error_msg}")
            return False

    # === Clear previous log ===
    with open(LOG_FILE, "w") as f:
        f.write("==== TPOT Wine Quality Pipeline Log ====\n")

    # === Training ===
    if full or train:
        if not task or not region:
            log("❌  Task and region are required for training.")
            return False
        retrain_script = (
            "scripts/retrain_classification.py" if task == "classification"
            else "scripts/retrain_regression.py"
        )
        result = run_subprocess(
            ["python", retrain_script, "--region", region],
            f"Training {task.capitalize()} Model"
        )
        if not result:
            return False

    # === Validation ===
    if full or validate:
        if not run_subprocess(["python", "scripts/validate_models.py", "--task", task], "Model Validation"):
            return False

    # === Notebook Export ===
    if full or notebooks:
        if not run_subprocess(["python", "scripts/export_notebooks.py", "--task", task], "Exporting Notebooks"):
            return False

    # === Report Generation ===
    if full or report:
        if not run_subprocess(["python", "scripts/validation_report.py", "--task", task], "Generating PDF Report"):
            return False

    # === Prediction ===
    if predict:
        if not all([model, data, output_dir]):
            log("❌  Prediction requires --model, --data, and --output_dir.")
            return False
        region_name = os.path.basename(model).split(".")[0].replace("combined_", "").replace("_filled", "")
        return run_subprocess([
            "python", "scripts/predict.py",
            "--region", region_name,
            "--data", data,
            "--output_dir", output_dir
        ], "Running Predictions")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["regression", "classification"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--notebooks", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--output_dir")
    parser.add_argument("--region")
    args = parser.parse_args()

    run_pipeline_backend(
        task=args.task,
        full=args.full,
        train=args.train,
        validate=args.validate,
        notebooks=args.notebooks,
        report=args.report,
        predict=args.predict,
        model=args.model,
        data=args.data,
        output_dir=args.output_dir,
        region=args.region
    )
