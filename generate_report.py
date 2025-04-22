from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

def fig_to_img(fig) -> BytesIO:
    """Convert a matplotlib figure to an in-memory image buffer."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_insight_report(
    regions: str,
    date_range: str,
    correlation_df: pd.DataFrame,
    scatter_fig,
    boxplot_fig,
    output_path: str = "reports/reportlab_report.pdf"
) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Wine Quality Insights Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Regions:</b> {regions}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date Range:</b> {date_range}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Correlation Table
    elements.append(Paragraph("<b>Top Correlated Features</b>", styles['Heading2']))
    elements.append(Spacer(1, 6))
    corr_html = correlation_df.to_html(index=True, border=0)
    elements.append(Paragraph(corr_html, styles['Code']))
    elements.append(Spacer(1, 12))

    # Images
    elements.append(Paragraph("Scatter Plot", styles['Heading2']))
    elements.append(Image(fig_to_img(scatter_fig), width=5*inch, height=3*inch))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Box Plot", styles['Heading2']))
    elements.append(Image(fig_to_img(boxplot_fig), width=5*inch, height=3*inch))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Generated with ❤️ by Baltzakis Themistoklis", styles['Normal']))
    doc.build(elements)

    return output_path

