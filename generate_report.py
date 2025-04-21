import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pathlib import Path

TEMPLATES_DIR = Path("templates")
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

def generate_wine_report(df: pd.DataFrame, target: str, report_name="wine_report.pdf"):
    # === Preprocess
    corr = df.corr(numeric_only=True)[target].drop(target).sort_values(key=abs, ascending=False).head(10)

    # === Plots
    scatter_path = EXPORT_DIR / "scatter_plot.png"
    boxplot_path = EXPORT_DIR / "box_plot.png"

    top_feat = corr.index[0]
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x=top_feat, y=target, hue="Region")
    plt.title(f"{top_feat} vs {target}")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x="Region", y=top_feat)
    plt.xticks(rotation=45)
    plt.title(f"{top_feat} Distribution by Region")
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()

    # === Prepare metadata
    regions = ", ".join(df["Region"].unique())
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    else:
        date_range = "N/A"

    # === HTML Rendering
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template("report_template.html")
    html_content = template.render(
        regions=regions,
        date_range=date_range,
        correlation_table=corr.to_frame(name="Correlation").to_html(),
        scatter_plot_path=scatter_path.resolve(),
        boxplot_path=boxplot_path.resolve()
    )

    pdf_path = EXPORT_DIR / report_name
    HTML(string=html_content).write_pdf(pdf_path)
    return pdf_path
