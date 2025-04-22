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
import qrcode

from kpi_descriptions import kpi_descriptions


def fig_to_img(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_kpi_summary(corr_df: pd.DataFrame) -> str:
    top_kpis = corr_df.copy()
    top_kpis["Correlation"] = pd.to_numeric(top_kpis["Correlation"], errors="coerce")
    top_kpis = top_kpis.dropna(subset=["Correlation"])
    top_kpis = top_kpis.reindex(top_kpis["Correlation"].abs().sort_values(ascending=False).index).head(3)

    summary_lines = []
    for idx, row in top_kpis.iterrows():
        desc = kpi_descriptions.get(idx, "N/A")
        direction = "positively" if row["Correlation"] > 0 else "negatively"
        summary_lines.append(
            f"• {idx.replace('_', ' ').title()} ({desc}) — {direction} correlated (r = {row['Correlation']:.3f})"
        )
    return "Key climate indicators influencing wine quality include:<br/>" + "<br/>".join(summary_lines)


def get_row_color(value: float):
    if value >= 0.7:
        return colors.lightgreen
    elif value >= 0.5:
        return colors.beige
    elif value < 0:
        return colors.pink
    return colors.white


def generate_insight_report(
    regions: str,
    date_range: str,
    correlation_df: pd.DataFrame,
    scatter_fig,
    boxplot_fig,
    metrics: dict,
    output_path: str = "reports/final_kpi_report.pdf",
    include_appendix: bool = False,
    dashboard_url: str = ""
) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("📘 <b>Wine Quality Insights Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Analyzed Regions:</b> {regions}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date Range:</b> {date_range}", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("📐 <b>Model Performance Metrics</b>", styles['Heading2']))
    elements.append(Paragraph(f"• R² Score: {metrics.get('r2', 0):.3f}", styles['Normal']))
    elements.append(Paragraph(f"• RMSE: {metrics.get('rmse', 0):.3f}", styles['Normal']))
    elements.append(Paragraph(f"• MAE: {metrics.get('mae', 0):.3f}", styles['Normal']))
    elements.append(Spacer(1, 12))

    corr_df_clean = correlation_df.dropna().copy()
    corr_df_clean["Correlation"] = pd.to_numeric(corr_df_clean["Correlation"], errors="coerce")
    corr_df_clean = corr_df_clean.dropna(subset=["Correlation"])
    corr_df_clean = corr_df_clean[abs(corr_df_clean["Correlation"]) >= 0.5]
    corr_df_clean = corr_df_clean.sort_values(by="Correlation", key=abs, ascending=False).head(20)
    corr_df_clean["Description"] = corr_df_clean.index.map(lambda x: kpi_descriptions.get(x, "N/A"))

    elements.append(Paragraph("📌 <b>Summary of Key Drivers</b>", styles['Heading2']))
    summary_text = generate_kpi_summary(corr_df_clean)
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("🔬 <b>Top Correlated Features (r ≥ 0.5)</b>", styles['Heading2']))
    table_data = [["Feature", "Correlation", "Description"]] + corr_df_clean.reset_index().values.tolist()
    table = Table(table_data, hAlign="LEFT", colWidths=[150, 80, 240])
    row_styles = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d1d1d1")),
                  ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                  ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                  ('FONTSIZE', (0, 0), (-1, -1), 8)]
    for i, row in enumerate(corr_df_clean.itertuples(), start=1):
        row_color = get_row_color(row.Correlation)
        row_styles.append(('BACKGROUND', (0, i), (-1, i), row_color))
    table.setStyle(TableStyle(row_styles))
    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(PageBreak())
    elements.append(Paragraph("📈 <b>Correlation Scatter Plot</b>", styles['Heading2']))
    elements.append(Image(fig_to_img(scatter_fig), width=5.5 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("📊 <b>KPI Box Plot</b>", styles['Heading2']))
    elements.append(Image(fig_to_img(boxplot_fig), width=5.5 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 20))

    elements.append(PageBreak())
    elements.append(Paragraph("📚 <b>4. Methodology</b>", styles['Heading2']))
    elements.append(Paragraph(
        "This report analyzes the correlation between meteorological variables and wine quality "
        "using Pearson correlation (r). Features with |r| ≥ 0.5 are considered significant. "
        "Scatter and box plots visualize relationships with the target variable.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("🔗 <b>5. References</b>", styles['Heading2']))
    elements.append(Paragraph(
        "• Baltzakis, T., 'Wine Quality Forecasting under Climate Variability', 2024<br/>"
        "• scikit-learn documentation<br/>"
        "• XGBoost documentation<br/>"
        "• ReportLab documentation",
        styles['Normal']
    ))

    if include_appendix:
        elements.append(PageBreak())
        elements.append(Paragraph("📎 <b>Appendix: Full Correlation Matrix</b>", styles['Heading2']))
        full_corr = correlation_df.dropna().copy()
        full_corr["Correlation"] = pd.to_numeric(full_corr["Correlation"], errors="coerce")
        full_corr = full_corr.dropna()
        appendix_data = [["Feature", "Correlation"]] + full_corr.reset_index().values.tolist()
        appendix_table = Table(appendix_data, colWidths=[300, 100], hAlign="LEFT")
        appendix_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(appendix_table)

    if dashboard_url:
        qr_img = qrcode.make(dashboard_url)
        qr_bytes = BytesIO()
        qr_img.save(qr_bytes, format="PNG")
        qr_bytes.seek(0)
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("🔗 Streamlit Dashboard Access", styles['Heading2']))
        elements.append(Image(qr_bytes, width=1.5 * inch, height=1.5 * inch))
        elements.append(Paragraph(dashboard_url, styles['Normal']))

    elements.append(Spacer(1, 30))
    elements.append(Paragraph("🧑‍🔬 Report generated by Baltzakis Themistoklis", styles['Normal']))
    elements.append(Paragraph(f"📅 Date: {date.today().isoformat()}", styles['Normal']))

    doc.build(elements)
    return output_path
