#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime
from scripts.utils.utils import generate_pdf_from_dict

# === Paths ===
LOG_DIR = "logs"
LEADERBOARD_FILES = {
    "Regression": os.path.join(LOG_DIR, "leaderboard_regression.csv"),
    "Classification": os.path.join(LOG_DIR, "leaderboard_classification.csv")
}
OUTPUT_PATH = os.path.join(LOG_DIR, "validation_report.pdf")

def collect_summary(df, task):
    summary = {
        "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Task Type": task,
        "Total Models": len(df)
    }
    if "accuracy" in df.columns:
        summary["Avg Accuracy"] = round(df["accuracy"].mean(), 4)
    if "r2" in df.columns:
        summary["Avg R²"] = round(df["r2"].mean(), 4)
    if "rmse" in df.columns:
        summary["Avg RMSE"] = round(df["rmse"].mean(), 4)

    sort_metric = "accuracy" if "accuracy" in df.columns else "r2"
    best = df.sort_values(by=sort_metric, ascending=False).iloc[0]
    summary["Top Region"] = best["region"]
    summary["Top Score"] = round(best.get(sort_metric, 0.0), 4)
    summary["Model Path"] = best.get("model_path", "N/A")

    return summary, df[["region", "accuracy", "r2", "rmse", "timestamp"]].fillna("").to_dict(orient="records")

def main():
    full_html = "<html><body><h1>Wine Quality Validation Report</h1>"

    for task_name, path in LEADERBOARD_FILES.items():
        if not os.path.exists(path):
            print(f"❌ No leaderboard found for {task_name}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"⚠️ Leaderboard for {task_name} is empty.")
            continue

        summary, table = collect_summary(df, task_name)
        html_section = f"<h2>{task_name} Summary</h2><ul>"
        for k, v in summary.items():
            html_section += f"<li><b>{k}:</b> {v}</li>"
        html_section += "</ul><h3>Model Performance</h3><table border='1'><tr>"

        if table:
            columns = table[0].keys()
            html_section += ''.join(f"<th>{col}</th>" for col in columns) + "</tr>"
            for row in table:
                html_section += "<tr>" + ''.join(f"<td>{row[col]}</td>" for col in columns) + "</tr>"
            html_section += "</table>"
        full_html += html_section

    full_html += "</body></html>"

    with open("tmp_validation.html", "w") as f:
        f.write(full_html)

    try:
        import pdfkit
        pdfkit.from_file("tmp_validation.html", OUTPUT_PATH)
        print(f"✅ Validation report saved to: {OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")

if __name__ == "__main__":
    main()
