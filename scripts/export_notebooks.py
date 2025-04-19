import os
import joblib
import pandas as pd
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from sklearn.metrics import r2_score, mean_squared_error
from utils import preprocess_data

MODELS_DIR = "models"
DATA_DIR = "data/processed_filled"
NOTEBOOKS_DIR = "notebooks"
R2_THRESHOLD = 0.7  # Only export notebooks for models that exceed this R²

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def extract_region_from_filename(filename):
    return filename.split("_")[0]

def evaluate_model(model, region):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"combined_{region}_filled.csv"))
        X, y = preprocess_data(df)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        return round(r2, 4), round(rmse, 4)
    except Exception as e:
        print(f"❌ Evaluation failed for {region}: {e}")
        return None, None

def create_notebook(region, pipeline_str, r2, rmse):
    nb = new_notebook(cells=[
        new_code_cell(f"# TPOT Exported Pipeline for {region}"),
        new_code_cell(f"# R² Score: {r2}\n# RMSE: {rmse}"),
        new_code_cell("import joblib\nimport pandas as pd\nfrom sklearn.metrics import r2_score, mean_squared_error"),
        new_code_cell(f"# Load your model\nmodel = joblib.load('models/{region}_<timestamp>.joblib')"),
        new_code_cell(f"# Load and preprocess your data\ndf = pd.read_csv('data/processed_filled/combined_{region}_filled.csv')\n# Preprocessing logic goes here..."),
        new_code_cell("# Run predictions and evaluate\n# X, y = preprocess_data(df)\n# y_pred = model.predict(X)\n# print('R²:', r2_score(y, y_pred))\n# print('RMSE:', mean_squared_error(y, y_pred, squared=False))"),
        new_code_cell(pipeline_str),
        new_code_cell("# Modify or extend the pipeline as needed...")
    ])
    nb_path = os.path.join(NOTEBOOKS_DIR, f"{region}_pipeline.ipynb")
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    print(f"✅ Notebook saved: {nb_path}")

def main():
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".joblib"):
            full_path = os.path.join(MODELS_DIR, file)
            region = extract_region_from_filename(file)

            try:
                model = joblib.load(full_path)
                r2, rmse = evaluate_model(model, region)

                if r2 is None:
                    continue

                if r2 >= R2_THRESHOLD:
                    pipeline_str = str(model)
                    create_notebook(region, pipeline_str, r2, rmse)
                else:
                    print(f"ℹ️  Skipping {region}: R² {r2} below threshold {R2_THRESHOLD}")

            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")

if __name__ == "__main__":
    main()
