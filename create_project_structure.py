import os
import json

# Define the folder and file structure
structure = {
    "XGBoost_Forecasting_App": {
        "models": ["model.pkl"],
        "data": {
            "processed_filled": ["combined_dataset.csv"],
            "": ["climate_lookup.json"]  # Files directly under "data/"
        },
        "assets": ["feature_importance.png", "predictions_vs_actuals.png"],
        "files": ["app.py", "schema.json", "metrics.json", "requirements.txt", "README.md"]
    }
}

def create_structure(base_path, structure):
    for root_folder, subfolders in structure.items():
        root_path = os.path.join(base_path, root_folder)
        os.makedirs(root_path, exist_ok=True)

        # Subfolders and their files
        for subfolder, files in subfolders.items():
            folder_path = os.path.join(root_path, subfolder)
            os.makedirs(folder_path, exist_ok=True)  # Ensure the subfolder exists

            for file in files:
                file_path = os.path.join(folder_path, file)
                with open(file_path, "w") as f:
                    if file.endswith(".json"):
                        json.dump({}, f)  # Create an empty JSON object
                    else:
                        f.write(f"# Placeholder for {file}\n")

        # Handle files directly under the root folder
        if "files" in subfolders:
            for file in subfolders["files"]:
                file_path = os.path.join(root_path, file)
                with open(file_path, "w") as f:
                    f.write(f"# Placeholder for {file}\n")

if __name__ == "__main__":
    base = os.getcwd()
    create_structure(base, structure)
    print("✅ Project structure created successfully!")
