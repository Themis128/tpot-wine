import os
import sys
import argparse
import warnings
import traceback
from pathlib import Path

import joblib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from sklearn.metrics import r2_score
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed_filled"
MODEL_DIR = ROOT_DIR / "models"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
TEMPLATE_NOTEBOOK = ROOT_DIR / "scripts" / "pipeline_template.ipynb"

NOTEBOOK_DIR.mkdir(exist_ok=True)

summary = {
    "evaluated": 0,
    "saved": 0,
    "skipped": [],
    "failed": []
}


def extract_region(filename):
    name = Path(filename).stem
    return name.split("_")[0] if name.startswith("combined") is False else None


def save_notebook(region):
    nb_path = NOTEBOOK_DIR / f"{region}_pipeline.ipynb"
    try:
        with open(TEMPLATE_NOTEBOOK) as f:
            nb = nbformat.read(f, as_version=4)

        # Replace placeholder with region
        for cell in nb.cells:
            if cell.cell_type == "code" and "# REGION_PLACEHOLDER" in cell.source:
                cell.source = cell.source.replace("# REGION_PLACEHOLDER", f"region = '{region}'")

        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
        ep.preprocess(nb, {'metadata': {'path': ROOT_DIR}})

        with open(nb_path, 'w') as f:
            nbformat.write(nb, f)

        print(f"✅    Notebook saved: {nb_path}")
        summary["saved"] += 1
    except Exception as e:
        print(f"❌    Failed to save notebook for {region}: {e}")
        summary["failed"].append((region, str(e)))
        traceback.print_exc()


def evaluate_and_generate(min_score, only_region):
    for model_file in MODEL_DIR.glob("*.joblib"):
        region = extract_region(model_file.name)
        if not region:
            continue
        if only_region and region.lower() != only_region.lower():
            continue

        summary["evaluated"] += 1
        model_path = MODEL_DIR / f"{region}_pipeline.joblib"
        data_path = DATA_DIR / f"{region}_filled.csv"

        if not model_path.exists() or not data_path.exists():
            summary["failed"].append((region, "Missing model or data"))
            continue

        try:
            model = joblib.load(model_path)
            df = pd.read_csv(data_path)
            X = df.drop(columns=["target"])
            y = df["target"]

            y_pred = model.predict(X)
            score = r2_score(y, y_pred)

            if score >= min_score:
                save_notebook(region)
            else:
                reason = f"R² {score:.4f} below threshold {min_score}"
                print(f"ℹ️  Skipping {region}: {reason}")
                summary["skipped"].append((region, reason))
        except ValueError as ve:
            if "feature names" in str(ve):
                msg = "Feature mismatch — check preprocessing"
            else:
                msg = str(ve)
            summary["failed"].append((region, msg))
            print(f"❌    Evaluation failed for {region}: {msg}")
        except Exception as e:
            summary["failed"].append((region, str(e)))
            print(f"❌    Evaluation failed for {region}: {e}")
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
    parser = argparse.ArgumentParser(description="Export region notebooks from trained TPOT pipelines.")
    parser.add_argument("--min-score", type=float, default=0.7, help="Minimum R² score to save a notebook (default: 0.7)")
    parser.add_argument("--only-region", type=str, help="Run for only this region (case-insensitive)")

    args = parser.parse_args()
    evaluate_and_generate(args.min_score, args.only_region)
    print_summary()
