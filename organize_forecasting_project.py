import os
import shutil
import json
from pathlib import Path

# === PATH SETUP ===
FORECASTING_DIR = Path("/mnt/c/Users/baltz/Desktop/Models/models/XGBoost/forecasting")
DATASETS_SRC = Path("/mnt/c/Users/baltz/Desktop/Models/processed_filled")
REGIONS = [f.name.split("_")[1] for f in DATASETS_SRC.glob("combined_*_filled.csv")]

# === CLIMATE LOOKUP GENERATION ===
def generate_climate_data(regions):
    import random
    return {
        region: {
            "mean_temp": round(random.uniform(19.0, 26.0), 1),
            "humidity": round(random.uniform(55.0, 75.0), 1)
        } for region in regions
    }

def organize_forecasting_dir():
    print(f"🚀 Organizing forecasting project in {FORECASTING_DIR}")

    # Create subfolders
    models_dir = FORECASTING_DIR / "models"
    data_dir = FORECASTING_DIR / "data"
    assets_dir = FORECASTING_DIR / "assets"

    for folder in [models_dir, data_dir, assets_dir]:
        folder.mkdir(exist_ok=True)

    # Move files into appropriate folders
    file_map = {
        "model.pkl": models_dir,
        "feature_importance.png": assets_dir,
        "predictions_vs_actuals.png": assets_dir,
        "schema.json": FORECASTING_DIR,
        "metrics.json": FORECASTING_DIR,
        "requirements.txt": FORECASTING_DIR,
        "README.md": FORECASTING_DIR,
        "app.py": FORECASTING_DIR
    }

    for file_name, target_dir in file_map.items():
        src = FORECASTING_DIR / file_name
        if src.exists():
            shutil.move(str(src), str(target_dir / file_name))
            print(f"✅ Moved {file_name} to {target_dir}")
        else:
            print(f"⚠️ {file_name} not found in {FORECASTING_DIR}")

    # Move wine datasets to /data
    print("📦 Moving regional wine datasets to /data/")
    for file in DATASETS_SRC.glob("combined_*_filled.csv"):
        shutil.copy2(file, data_dir)

    # Generate climate_lookup.json
    print("🌤️ Generating climate_lookup.json")
    climate_data = generate_climate_data(REGIONS)
    with open(data_dir / "climate_lookup.json", "w") as f:
        json.dump(climate_data, f, indent=4)

    print("✅ Reorganization complete!")

if __name__ == "__main__":
    organize_forecasting_dir()
