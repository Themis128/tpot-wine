from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import date
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd


def fig_to_img(fig) -> BytesIO:
    """Convert a matplotlib figure to an in-memory image buffer."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_kpi_summary(corr_df: pd.DataFrame) -> str:
    """Generate a summary paragraph of the top 3 correlated KPIs."""
    top_kpis = corr_df.dropna().abs().sort_values(by="Correlation", ascending=False).head(3)
    summary_lines = []
    for idx, row in top_kpis.iterrows():
        direction = "positively" if row["Correlation"] > 0 else "negatively"
        summary_lines.append(f"• {idx.replace('_', ' ').title()} — {direction} correlated (r = {row['Correlation']:.3f})")
    return "Key climate indicators influencing wine quality include:\n" + "\n".join(summary_lines)


def generate_insight_report(
    regions: str,
    date_range: str,
    correlation_df: pd.DataFrame,
    scatter_fig,
    boxplot_fig,
    metrics: dict,
    output_path: str = "reports/final_insight_report.pdf",
    include_appendix: bool = False
) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("📘 <b>Wine Quality Insights Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Analyzed Regions:</b> {regions}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date Range:</b> {date_range}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Model Metrics
    elements.append(Paragraph("📐 <b>Model Performance Metrics</b>", styles['Heading2']))
    elements.append(Paragraph(f"• R² Score: {metrics.get('r2', 0):.3f}", styles['Normal']))
    elements.append(Paragraph(f"• RMSE: {metrics.get('rmse', 0):.3f}", styles['Normal']))
    elements.append(Paragraph(f"• MAE: {metrics.get('mae', 0):.3f}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Filter for KPIs
    corr_df_clean = correlation_df.dropna().round(3)
    corr_df_clean = corr_df_clean[abs(corr_df_clean["Correlation"]) >= 0.5]
    corr_df_clean = corr_df_clean.sort_values(by="Correlation", key=abs, ascending=False).head(20)

    # Summary
    elements.append(Paragraph("📌 <b>Summary of Key Drivers</b>", styles['Heading2']))
    summary_text = generate_kpi_summary(corr_df_clean)
    elements.append(Paragraph(summary_text.replace("\n", "<br/>"), styles['Normal']))
    elements.append(Spacer(1, 12))

    # Table
    elements.append(Paragraph("🔬 <b>Top Correlated Features (r ≥ 0.5)</b>", styles['Heading2']))
    table_data = [["Feature", "Correlation"]] + corr_df_clean.reset_index().values.tolist()
    table = Table(table_data, hAlign="LEFT", colWidths=[300, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d1d1d1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Visuals
    elements.append(PageBreak())
    elements.append(Paragraph("📈 <b>Correlation Scatter Plot</b>", styles['Heading2']))
    elements.append(Image(fig_to_img(scatter_fig), width=5.5 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("📊 <b>KPI Box Plot</b>", styles['Heading2']))
    elements.append(Image(fig_to_img(boxplot_fig), width=5.5 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 20))

    # Methodology & References
    elements.append(PageBreak())
    elements.append(Paragraph("📚 <b>4. Methodology</b>", styles['Heading2']))
    elements.append(Paragraph(
        "This report is based on correlation analysis between meteorological variables and "
        "wine quality scores collected from regional vineyards. Variables with absolute Pearson correlation ≥ 0.5 "
        "were considered significant. Visual analysis includes scatter and box plots for top predictors. "
        "Model performance is assessed using R², RMSE, and MAE.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("🔗 <b>5. References</b>", styles['Heading2']))
    elements.append(Paragraph(
        "• Baltzakis, T. et al., 'Wine Quality Forecasting under Climate Variability', 2024<br/>"
        "• Scikit-learn documentation — https://scikit-learn.org<br/>"
        "• XGBoost documentation — https://xgboost.readthedocs.io<br/>"
        "• ReportLab — https://www.reportlab.com/",
        styles['Normal']
    ))
    elements.append(Spacer(1, 24))

    # Optional Appendix
    if include_appendix:
        elements.append(PageBreak())
        elements.append(Paragraph("📎 <b>Appendix: Full Correlation Matrix</b>", styles['Heading2']))
        full_corr = correlation_df.dropna().round(3).sort_values(by="Correlation", key=abs, ascending=False)
        table_data_full = [["Feature", "Correlation"]] + full_corr.reset_index().values.tolist()
        appendix_table = Table(table_data_full, hAlign="LEFT", colWidths=[300, 100])
        appendix_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        elements.append(appendix_table)

    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(f"🧑‍🔬 Report generated by Baltzakis Themistoklis", styles['Normal']))
    elements.append(Paragraph(f"📅 Date: {date.today().isoformat()}", styles['Normal']))

    doc.build(elements)
    return output_path
