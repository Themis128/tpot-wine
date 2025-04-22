from jinja2 import Environment, FileSystemLoader
import pdfkit
from pathlib import Path
import pandas as pd

def generate_html_table(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame into an HTML table for the report."""
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
    """Generate a PDF insight report from a template and context."""

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    # Render context for HTML
    context = {
        "regions": regions,
        "date_range": date_range,
        "correlation_table": generate_html_table(correlation_df),
        "scatter_plot_path": Path(scatter_plot_path).resolve(),
        "boxplot_path": Path(boxplot_path).resolve()
    }

    # Render the HTML template
    html_content = template.render(context)

    # Render PDF from HTML
    pdfkit.from_string(html_content, output_path)

    return str(output_path)
