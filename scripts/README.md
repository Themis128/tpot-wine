Sure! Here's a clean and complete README.md that explains how the Streamlit Dashboard for the TPOT Wine Quality project works, including setup, usage, and features:


---

# Greek Wine Quality Dashboard

A **Streamlit** application for visualizing, training, and predicting wine quality across Greek regions using **AutoML (TPOT)** pipelines.

---

## Features

- **Leaderboard**: Compare models by accuracy, R² or RMSE.
- **Climate Simulation**: See how temperature & humidity affect wine parameters.
- **Wine Prediction**: Predict wine quality using real samples or manual inputs.
- **Reports**: Generate and download PDF validation summaries.

---

## Directory Structure

tpot-wine-quality/ │ ├── streamlit-app.py           # Main Streamlit dashboard ├── scripts/                   # All backend logic │   ├── train_model_unified.py │   ├── retrain-app.py │   ├── run_pipeline.py │   ├── predict.py │   ├── validate_models.py │   ├── validation_report.py │   └── utils/ │       ├── preprocessing.py │       ├── io.py │       └── utils_pdf.py ├── models/                    # Saved .joblib models ├── logs/                      # Leaderboard CSV + log files ├── data/processed_filled/     # Cleaned datasets per region

---

## Setup

1. **Create & activate virtual environment** (optional):
```bash
python3 -m venv venv
source venv/bin/activate

2. Install requirements:



pip install -r requirements.txt

3. Run the Streamlit app:



streamlit run streamlit-app.py


---

How It Works

Tab 1: Leaderboard

Loads logs/leaderboard.csv

Sorts and displays models by performance metric

You can inspect model pipelines per region

Click "Run Full AutoML Pipeline" to trigger training, validation, and notebook generation (scripts/run_pipeline.py)


Tab 2: Climate Simulation

Uses quadratic response functions to simulate how mean temperature and humidity affect:

Alcohol Strength (TAS)

Residual Sugars

Sulfites


Contour plots show parameter sensitivity

Export results as CSV or PDF


Tab 3: Wine Prediction

Uses the most recent model for selected region

Two prediction modes:

From dataset sample row

Manual input of wine attributes


Predictions can be exported as CSV or PDF


Tab 4: Reports

Download the latest leaderboard.csv

View bar charts for performance comparison

Generate a Validation Summary PDF Report (scripts/validation_report.py)



---

Advanced Usage

To run backend scripts manually:

Run Full Pipeline

python scripts/run_pipeline.py --task regression --full

Train Models Only

python scripts/run_pipeline.py --task regression --train

Predict from terminal

python scripts/predict.py --model models/YourModel.joblib --data data/your_data.csv


---

Requirements

Python 3.8+

TPOT

Streamlit

Scikit-learn

Pandas

NumPy

Matplotlib

imbalanced-learn

pdfkit + wkhtmltopdf



---

Credits

Developed by: [Your Name / Team]
Inspired by regional climate impact on wine quality.


---

Let me know if you'd like me to generate this file for you, or customize the project name or instructions!


