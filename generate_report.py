from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pathlib import Path
import pandas as pd

def generate_html_table(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame into an HTML table."""
    return df.to_html(classes='dataframe', border=0, index=True)

def generate_insight_report(
    regions: str,
    date_range: str,
    correlation_df: pd.DataFrame,
    scatter_plot_path: str,
    boxplot_path: str,
    template_dir: str = "templates",
    template_name: str = "report_template.html",
    output_path: str = "reports/insight_report.pdf"
) -> str:
    """Generate a PDF report using HTML template rendering + WeasyPrint."""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load Jinja2 template
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    # Prepare the HTML with dynamic content
    context = {
        "regions": regions,
        "date_range": date_range,
        "correlation_table": generate_html_table(correlation_df),
        "scatter_plot_path": Path(scatter_plot_path).resolve().as_uri(),
        "boxplot_path": Path(boxplot_path).resolve().as_uri()
    }

    html_content = template.render(context)

    # Convert HTML to PDF
    HTML(string=html_content).write_pdf(output_path)

    return str(output_path)
