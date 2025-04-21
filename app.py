import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import re
from generate_report import generate_wine_report

# === PATHS ===
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# === LOAD MODEL & METADATA ===
model = joblib.load(MODELS_DIR / "model.pkl")
with open(BASE_DIR / "metrics.json") as f: metrics = json.load(f)
with open(BASE_DIR / "schema.json") as f: schema = json.load(f)
with open(DATA_DIR / "climate_lookup.json") as f: climate_data = json.load(f)

# === FEATURE NAME CLEANER ===
def prettify(name):
    name = name.replace("_", " ")
    name = re.sub(r'\btemp\b', 'Temperature', name, flags=re.IGNORECASE)
    name = re.sub(r'\bhum\b', 'Humidity', name, flags=re.IGNORECASE)
    name = re.sub(r'\bp\b', 'Precipitation', name)
    name = name.replace("avg", "Average").replace("mean", "Mean").replace("sum", "Sum")
    return name.title()

# === REGION + DATE INFO DISPLAY ===
def get_region_date_summary(df: pd.DataFrame) -> str:
    regions = df["Region"].unique().tolist() if "Region" in df.columns else ["Unknown"]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        start_date = df["date"].min().date()
        end_date = df["date"].max().date()
        date_range = f"{start_date} to {end_date}"
    else:
        date_range = "N/A"
    return f"📍 **Regions:** {', '.join(regions)}  \n🗓️ **Date Range:** {date_range}"

# === PAGE CONFIG ===
st.set_page_config(page_title="🍷 Wine Forecast Dashboard", layout="wide")
st.title("🍷 Wine Quality & Climate Forecast")

# === SIDEBAR NAVIGATION ===
section = st.sidebar.radio("📚 Choose Section", [
    "🗂️ Project Overview", "📂 Explore Datasets", "🌤️ Climate Impact",
    "🔮 Predict One Sample", "📁 Predict Multiple Samples",
    "📊 Model Evaluation", "📊 Advanced Analytics", "💾 Export Tools"
])

# === 1. OVERVIEW ===
if section == "��️ Project Overview":
    st.subheader("🔍 Project Summary")
    st.markdown("""
    This app forecasts wine quality using vineyard weather data and machine learning.
    Explore datasets, simulate climate scenarios, and make predictions interactively.
    """)
    st.markdown("### 📊 Model Performance")
    st.json(metrics)
    st.markdown("---")
    st.markdown("📌 Created by **Baltzakis Themistoklis**")

# === 2. DATASET EXPLORER ===
elif section == "📂 Explore Datasets":
    st.subheader("📂 Explore Wine Region Datasets")

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith("climate")]
    regions = [f.replace("combined_", "").replace("_filled.csv", "") for f in files]
    selected = st.multiselect("Select wine regions", regions)

    dfs = []
    for region in selected:
        df = pd.read_csv(DATA_DIR / f"combined_{region}_filled.csv")
        df["Region"] = region
        dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        st.session_state["combined_df"] = combined_df
        st.markdown(get_region_date_summary(combined_df))
        st.dataframe(combined_df.head())
        st.markdown("### 📈 Summary Statistics")
        st.dataframe(combined_df.describe())
    else:
        st.info("Select one or more regions to view data.")

# === 3. CLIMATE SIMULATOR ===
elif section == "🌤️ Climate Impact":
    st.subheader("🌤️ Climate Scenario Simulator")

    region = st.selectbox("Choose a wine region", sorted(climate_data.keys()))
    default = climate_data[region]
    temp = st.slider("Mean Temperature (°C)", 15.0, 30.0, default["mean_temp"])
    humidity = st.slider("Relative Humidity (%)", 40.0, 90.0, default["humidity"])

    st.markdown("#### 🔎 Climate Impact Insight")
    if temp > 24 and humidity < 60:
        st.success("🔥 Higher sugar & alcohol expected due to heat and dryness.")
    elif temp < 21 and humidity > 70:
        st.warning("❄️ Potential drop in alcohol, more acidity.")
    else:
        st.info("🌿 Balanced climate — moderate impact expected.")

# === 4. SINGLE PREDICTION ===
elif section == "🔮 Predict One Sample":
    st.subheader("🔮 Manual Input Wine Quality Prediction")

    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(schema["features"]):
        label = prettify(feat)
        inputs[feat] = cols[i % 3].number_input(label, value=0.0)

    if st.button("Predict Quality"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Quality Score: **{round(prediction, 2)}**")

# === 5. BATCH PREDICTIONS ===
elif section == "📁 Predict Multiple Samples":
    st.subheader("📁 Batch Predict from CSV")

    file = st.file_uploader("Upload CSV with feature columns", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = [col for col in schema["features"] if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["Predicted_Quality"] = model.predict(df)
            st.success("✅ Predictions generated!")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("�� Download Predictions CSV", csv, "wine_predictions.csv")

# === 6. MODEL PERFORMANCE ===
elif section == "📊 Model Evaluation":
    st.subheader("📊 Model Visuals")
    st.image(ASSETS_DIR / "predictions_vs_actuals.png", caption="Predicted vs Actual", use_container_width=True)
    st.image(ASSETS_DIR / "feature_importance.png", caption="Feature Importance", use_container_width=True)

# === 7. ADVANCED ANALYTICS ===
elif section == "📊 Advanced Analytics":
    st.subheader("📊 Key Performance Insights")

    if "combined_df" not in st.session_state:
        st.warning("Please load regions in 'Explore Datasets' first.")
    else:
        df = st.session_state["combined_df"]
        st.markdown(get_region_date_summary(df))

        target = schema["target"]
        corr_matrix = df.corr(numeric_only=True)

        if target not in corr_matrix.columns:
            st.error(f"Target column '{target}' not found.")
        else:
            target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False).head(10)

            st.markdown("### 🔝 Top Features Correlated with Wine Quality")
            st.dataframe(
                target_corr.to_frame(name="Correlation with Quality")
                .style.background_gradient(cmap="coolwarm")
            )

            st.markdown("### 📈 Interactive Feature Exploration")
            selected_feature = st.selectbox("Choose a feature to explore", target_corr.index.tolist())

            # === Scatter + trendline
            fig = px.scatter(df, x=selected_feature, y=target, color="Region",
                             title=f"{selected_feature} vs {target}", trendline="ols", template="plotly_white")
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)

            # === Boxplot
            st.markdown("### 📦 Distribution by Region")
            fig2 = px.box(df, x="Region", y=selected_feature, color="Region",
                          title=f"{selected_feature} by Region", template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

            # === PDF Report Download
            st.markdown("### 📄 Export as PDF")
            if st.button("📥 Generate PDF Report"):
                pdf_path = generate_wine_report(df, target)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download PDF Report", f.read(), file_name=pdf_path.name, mime="application/pdf")

# === 8. EXPORT TOOLS ===
elif section == "💾 Export Tools":
    st.subheader("💾 Export Datasets and Insights")

    if "combined_df" in st.session_state:
        df = st.session_state["combined_df"]
        st.markdown(get_region_date_summary(df))

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Current Dataset", csv_data, "filtered_wine_dataset.csv")

        target = schema["target"]
        corr_target = df.corr(numeric_only=True)[target].sort_values(ascending=False)
        corr_out = corr_target.drop(labels=[target]).to_frame(name="Correlation")
        st.download_button("📥 Download Feature Correlations", corr_out.to_csv().encode("utf-8"), "feature_target_correlations.csv")

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.markdown("#### 👨‍🔬 App by Baltzakis Themistoklis")
