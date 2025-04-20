# export_notebooks_automl.py (refactored for JARML multi-backend)

import os
import argparse
import traceback
from pathlib import Path
from log_utils import setup_logger

import joblib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from sklearn.metrics import r2_score
import pandas as pd

logger = setup_logger()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed_filled"
MODEL_DIR = ROOT_DIR / "models"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
TEMPLATE_NOTEBOOK = ROOT_DIR / "scripts" / "pipeline_template.ipynb"

NOTEBOOK_DIR.mkdir(exist_ok=True)

summary = {"evaluated": 0, "saved": 0, "skipped": [], "failed": []}

def extract_region(filename):
    parts = Path(filename).stem.split("_")
    return parts[1] if len(parts) > 1 else None

def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Unable to load model: {e}")

def save_notebook(region):
    nb_path = NOTEBOOK_DIR / f"{region}_pipeline.ipynb"
    try:
        with open(TEMPLATE_NOTEBOOK) as f:
            nb = nbformat.read(f, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code" and "# REGION_PLACEHOLDER" in cell.source:
                cell.source = cell.source.replace("# REGION_PLACEHOLDER", f"region = '{region}'")
        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
        ep.preprocess(nb, {'metadata': {'path': ROOT_DIR}})
        with open(nb_path, 'w') as f:
            nbformat.write(nb, f)
        print(f"✅ Notebook saved: {nb_path}")
        summary["saved"] += 1
    except Exception as e:
        print(f"❌ Failed to save notebook for {region}: {e}")
        summary["failed"].append((region, str(e)))
        traceback.print_exc()

def evaluate_and_generate(min_score=0.7, only_region=None):
    for model_file in MODEL_DIR.glob("*.joblib"):
        region = extract_region(model_file.name)
        if not region or (only_region and region.lower() != only_region.lower()):
            continue

        summary["evaluated"] += 1
        model_path = model_file
        data_path = DATA_DIR / f"combined_{region}_filled.csv"

        if not model_path.exists() or not data_path.exists():
            summary["failed"].append((region, "Missing model or data"))
            continue

        try:
            model = load_model(model_path)
            df = pd.read_csv(data_path)
            y = df["wine_quality_score"]
            X = df.drop(columns=["date", "region", "wine_quality_score"], errors="ignore").fillna(0)
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)

            if score >= min_score:
                save_notebook(region)
            else:
                reason = f"R² {score:.4f} below threshold {min_score}"
                print(f"ℹ️ Skipping {region}: {reason}")
                summary["skipped"].append((region, reason))
        except Exception as e:
            summary["failed"].append((region, str(e)))
            print(f"❌ Evaluation failed for {region}: {e}")
            traceback.print_exc()

def print_summary():
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Evaluated: {summary['evaluated']}")
    print(f"Saved:     {summary['saved']}")
    print(f"Skipped:   {len(summary['skipped'])}")
    for region, reason in summary["skipped"]:
        print(f"  - {region}: {reason}")
    print(f"Failed:    {len(summary['failed'])}")
    for region, error in summary["failed"]:
        print(f"  - {region}: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export notebooks from trained AutoML models")
    parser.add_argument("--min-score", type=float, default=0.7, help="Minimum R² score threshold")
    parser.add_argument("--only-region", type=str, help="Run only for a specific region")
    args = parser.parse_args()

    evaluate_and_generate(min_score=args.min_score, only_region=args.only_region)
    print_summary()
