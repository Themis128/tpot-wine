Here’s a focused README.md section specifically for the export_notebooks.py script:


---

export_notebooks.py

This script evaluates trained TPOT pipelines and generates executed Jupyter notebooks for regions that meet a configurable R² threshold. It can also run for a single region and provides a summary of the outcome.


---

Usage

python scripts/export_notebooks.py [OPTIONS]


---

Options


---

Example Commands

Export notebooks for all regions with R² ≥ 0.7:

python scripts/export_notebooks.py

Export for a single region:

python scripts/export_notebooks.py --only-region Samos

Lower the minimum score threshold:

python scripts/export_notebooks.py --min-score 0.5



---

Outputs

Saved notebooks are written to:

notebooks/<Region>_pipeline.ipynb

The script prints a summary at the end:

Total regions evaluated

Notebooks saved

Regions skipped (with reason)

Any failures (e.g., missing data or model, runtime errors)




---

Requirements

Ensure the following are available:

Trained TPOT models in models/

Processed data CSVs in data/processed_filled/

Notebook template at scripts/pipeline_template.ipynb



---

Let me know if you'd like this wrapped into a full project-level README or broken into docs/ format.


