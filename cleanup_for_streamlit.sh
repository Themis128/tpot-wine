#!/bin/bash

PROJECT_DIR="$HOME/tpot-wine"
BACKUP_DIR="$HOME/tpot-wine-backup"
FILES_AND_DIRS_TO_REMOVE=(
  "notebooks"
  "batch_predictions"
  "dashboard.py"
  "model_report.csv"
  "report_models.py"
  "train_model_unified.py"
  "README.md"
)

cd "$PROJECT_DIR" || {
    echo "Error: $PROJECT_DIR does not exist."
    exit 1
}

# === Dry Run Mode ===
if [[ $1 == "--dry-run" ]]; then
    echo "Dry Run Mode:"
    echo "The following files/directories would be removed:"
    for item in "${FILES_AND_DIRS_TO_REMOVE[@]}"; do
        [[ -e $item ]] && echo "  - $item"
    done
    exit 0
fi

# === Reset from Backup ===
if [[ $1 == "--reset" ]]; then
    if [ -d "$BACKUP_DIR" ]; then
        echo "Restoring backup from $BACKUP_DIR..."
        cp -r "$BACKUP_DIR"/* "$PROJECT_DIR"/
        echo "Restore complete."
    else
        echo "No backup directory found at $BACKUP_DIR."
    fi
    exit 0
fi

# === Confirm Before Deleting ===
echo "This will clean up unnecessary files for Streamlit."
read -p "Are you sure you want to continue? (y/N): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Aborted."
    exit 0
fi

# === Create Backup Directory ===
mkdir -p "$BACKUP_DIR"

# === Process Files ===
COUNT=0
for item in "${FILES_AND_DIRS_TO_REMOVE[@]}"; do
    if [ -e "$item" ]; then
        echo "Backing up and removing: $item"
        mv "$item" "$BACKUP_DIR"/
        ((COUNT++))
    else
        echo "Skipping (not found): $item"
    fi
done

echo "Cleanup complete. $COUNT items moved to $BACKUP_DIR."
