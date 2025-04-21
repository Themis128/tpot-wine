import os
import shutil

# Define folder structure
structure = {
    "app": [],
    "models/version_latest": ["random_forest_readable.pkl", "feature_names_readable.txt", "model_timestamp.txt"],
    "data": ["cleaned_combined_data.csv"],
    "scripts": ["refit_model_with_readable_names.py"],
    "assets": [
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "learning_curves.png", "roc_curve.png", "precision_recall_curve.png"
    ],
    "logs": [f for f in os.listdir() if f.startswith("learner_fold_") and f.endswith(".log")],
    "predictions": ["predictions_out_of_folds.csv"],
    ".": ["README.md", "framework.json", "status.txt"]  # keep in root
}

# Create folders & move files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for filename in files:
        if os.path.exists(filename):
            shutil.move(filename, os.path.join(folder, os.path.basename(filename)))
            print(f"✅ Moved {filename} -> {folder}/")
        else:
            print(f"⚠️ File not found: {filename}")

# Rename model + features for consistency
model_latest = "models/version_latest"
os.rename(f"{model_latest}/random_forest_readable.pkl", f"{model_latest}/model.pkl")
os.rename(f"{model_latest}/feature_names_readable.txt", f"{model_latest}/feature_names.txt")

print("\n✅ Project structure updated!")
