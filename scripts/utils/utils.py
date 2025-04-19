import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import pdfkit

# ===================================
# Data Preprocessing Utilities
# ===================================

def preprocess_data(df):
    y = df["wine_quality_score"]
    X = df.drop(columns=["wine_quality_score", "date"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.std() > 0]
    X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

    if y.dtype == "int" or y.nunique() < 20:
        valid_classes = y.value_counts()[lambda x: x >= 2].index
        mask = y.isin(valid_classes)
        return X[mask], y[mask]
    return X, y

def split_and_resample(X, y):
    stratify = y if len(np.unique(y)) < 20 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, test_size=0.25, random_state=42)
    if y.dtype == "int" or len(np.unique(y)) < 20:
        min_class_size = min(Counter(y_train).values())
        k_neighbors = max(1, min(min_class_size - 1, 5))
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train, X_test, y_test

# ===================================
# Model Saving Utility (with versioning)
# ===================================

def save_model(pipeline, model_dir, region, versioned=True, update_latest=True):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"combined_{region}_filled_{timestamp}.joblib" if versioned else f"{region}.joblib"
    model_path = os.path.join(model_dir, filename)
    joblib.dump(pipeline, model_path)

    if update_latest:
        latest_dir = os.path.join(model_dir, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        latest_link = os.path.join(latest_dir, f"{region}.joblib")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.abspath(model_path), latest_link)
    return model_path

# ===================================
# PDF Report Generation
# ===================================

def generate_pdf_from_dict(title, data_dict, filename="report.pdf", table_data=None, table_title=None):
    html = f"""
    <html>
    <head>
        <style>
            h1 {{
                color: #4CAF50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                text-align: left;
                padding: 8px;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{ background-color: #f2f2f2 }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <table>
    """
    for key, value in data_dict.items():
        html += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"
    html += "</table>"

    if table_data:
        html += f"<h2>{table_title or 'Details'}</h2><table>"
        if table_data:
            headers = table_data[0].keys()
            html += "<tr>" + "".join(f"<th>{col}</th>" for col in headers) + "</tr>"
            for row in table_data:
                html += "<tr>" + "".join(f"<td>{row.get(col, '')}</td>" for col in headers) + "</tr>"
        html += "</table>"

    html += "</body></html>"
    pdfkit.from_string(html, filename)
    return os.path.abspath(filename)

def generate_combined_validation_pdf(leaderboard_paths_dict, output_file="combined_report.pdf"):
    summary_data = {}
    all_tables = {}

    for label, path in leaderboard_paths_dict.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue

        summary_data[f"{label} - Total Models"] = len(df)

        if "accuracy" in df.columns:
            summary_data[f"{label} - Avg Accuracy"] = round(df["accuracy"].mean(), 4)
        if "r2" in df.columns:
            summary_data[f"{label} - Avg R²"] = round(df["r2"].mean(), 4)
        if "rmse" in df.columns:
            summary_data[f"{label} - Avg RMSE"] = round(df["rmse"].mean(), 4)

        sort_metric = "accuracy" if "accuracy" in df.columns else "r2"
        top = df.sort_values(by=sort_metric, ascending=False).iloc[0]
        summary_data[f"{label} - Top Region"] = top["region"]
        summary_data[f"{label} - Top Score"] = round(top.get(sort_metric, 0.0), 4)

        table_cols = [c for c in ["region", "accuracy", "r2", "rmse", "timestamp"] if c in df.columns]
        all_tables[label] = df[table_cols].fillna("").to_dict(orient="records")

    # Flatten the tables into one section
    combined_table = []
    for section, records in all_tables.items():
        for row in records:
            row["Model Type"] = section
            combined_table.append(row)

    return generate_pdf_from_dict(
        title="Combined Validation Report",
        data_dict=summary_data,
        filename=output_file,
        table_data=combined_table,
        table_title="All Model Results"
    )
