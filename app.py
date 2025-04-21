import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from PIL import Image
from generate_report import generate_wine_report

# === PATHS ===
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# === LOAD ASSETS ===
model = joblib.load(MODELS_DIR / "model.pkl")
with open(BASE_DIR / "metrics.json") as f: metrics = json.load(f)
with open(BASE_DIR / "schema.json") as f: schema = json.load(f)
with open(DATA_DIR / "climate_lookup.json") as f: climate_data = json.load(f)

# === PAGE SETUP ===
st.set_page_config(page_title="Wine Quality Forecast", layout="wide")
st.title("Wine Quality & Climate Forecast")

# === SIDEBAR NAV ===
section = st.sidebar.radio("Navigation", [
    "✦ Project Overview",
    "⟶ Explore Datasets",
    "Δ Climate Simulator",
    "ƒ(x) Predict One Sample",
    "⤵ Batch Predictions",
    "✔ Model Evaluation",
    "Σ Advanced Analytics",
    "⇩ Export Tools"
])

st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ _Hover over each section to learn more_")
st.sidebar.markdown("👨‍�� App by Baltzakis Themistoklis")

# === CLEAN FEATURE NAMES ===
def prettify(name):
    name = name.replace("_", " ")
    name = re.sub(r'\btemp\b', 'Temperature', name, flags=re.IGNORECASE)
    name = re.sub(r'\bhum\b', 'Humidity', name, flags=re.IGNORECASE)
    name = re.sub(r'\bp\b', 'Precipitation', name)
    name = name.replace("avg", "Average").replace("mean", "Mean").replace("sum", "Sum")
    return name.title()

# === 1. OVERVIEW ===
if section == "✦ Project Overview":
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("✦ Project Overview")
        st.markdown("""
        This platform leverages machine learning to forecast wine quality  
        using climate indicators and chemical properties.
        """)
    with col2:
        st.image(str(ASSETS_DIR / "logo.png"), width=180)

    st.markdown("---")
    st.markdown("### ✔ Model KPIs")

    perf_cols = st.columns(3)
    perf_cols[0].metric("📐 RMSE", f"{metrics.get('rmse', 'N/A'):.3f}")
    perf_cols[1].metric("📈 R² Score", f"{metrics.get('r2', 'N/A'):.3f}")
    perf_cols[2].metric("📊 MAE", f"{metrics.get('mae', 'N/A'):.3f}")

    st.markdown("---")
    st.markdown("✅ Explore data, run predictions, export PDFs and reach data-driven vineyard perfection.")

# === 2. EXPLORE DATASETS ===
elif section == "⟶ Explore Datasets":
    st.subheader("⟶ Explore Datasets")
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
        st.dataframe(combined_df.head())
        st.markdown("### Summary Statistics")
        st.dataframe(combined_df.describe())
    else:
        st.info("Select one or more regions to view data.")

# === 3. CLIMATE SIMULATOR ===
elif section == "Δ Climate Simulator":
    st.subheader("Δ Climate Impact Simulator")
    region = st.selectbox("Choose a wine region", sorted(climate_data.keys()))
    default = climate_data[region]
    temp = st.slider("Mean Temperature (°C)", 15.0, 30.0, default["mean_temp"])
    humidity = st.slider("Relative Humidity (%)", 40.0, 90.0, default["humidity"])

    st.markdown("#### Impact Summary")
    if temp > 24 and humidity < 60:
        st.success("Hot & dry → High sugar/alcohol expected.")
    elif temp < 21 and humidity > 70:
        st.warning("Cool & wet → Higher acidity, less alcohol.")
    else:
        st.info("Balanced conditions.")

# === 4. SINGLE PREDICTION ===
elif section == "ƒ(x) Predict One Sample":
    st.subheader("ƒ(x) Predict One Sample")
    inputs = {}
    cols = st.columns(3)
    for i, feat in enumerate(schema["features"]):
        label = prettify(feat)
        inputs[feat] = cols[i % 3].number_input(label, value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Wine Quality Score: **{round(prediction, 2)}**")

# === 5. BATCH PREDICTION ===
elif section == "⤵ Batch Predictions":
    st.subheader("⤵ Predict from CSV")
    file = st.file_uploader("Upload your CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = [col for col in schema["features"] if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["Predicted_Quality"] = model.predict(df)
            st.success("Prediction successful!")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "predictions.csv")

# === 6. MODEL EVAL ===
elif section == "✔ Model Evaluation":
    st.subheader("✔ Model Visuals")
    st.image(ASSETS_DIR / "predictions_vs_actuals.png", caption="Prediction vs Actual", use_container_width=True)
    st.image(ASSETS_DIR / "feature_importance.png", caption="Feature Importance", use_container_width=True)

# === 7. ADVANCED ANALYTICS ===
elif section == "Σ Advanced Analytics":
    st.subheader("Σ Feature Insights")
    if "combined_df" not in st.session_state:
        st.warning("Explore datasets first.")
    else:
        df = st.session_state["combined_df"]
        target = schema["target"]
        tab1, tab2 = st.tabs(["Top KPIs", "KPI vs Target Plot"])

        with tab1:
            corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
            kpis = corr.drop(target).head(10)
            st.dataframe(kpis.to_frame(name="Correlation").style.background_gradient(cmap="coolwarm"))

        with tab2:
            top_feat = st.selectbox("Select KPI", kpis.index.tolist())
            fig = px.scatter(df, x=top_feat, y=target, color="Region", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

# === 8. EXPORT ===
elif section == "⇩ Export Tools":
    st.subheader("⇩ Export Reports and Templates")
    if "combined_df" in st.session_state:
        df = st.session_state["combined_df"]
        st.download_button("Download Current Dataset", df.to_csv(index=False), "wine_data.csv")
        template = pd.DataFrame([{feat: 0.0 for feat in schema["features"]}])
        st.download_button("Download Input Template", template.to_csv(index=False), "input_template.csv")
        if st.button("Generate PDF Report"):
            report_path = generate_wine_report(df, BASE_DIR / "templates" / "report_template.html")
            with open(report_path, "rb") as f:
                st.download_button("Download PDF Report", f.read(), file_name="wine_report.pdf", mime="application/pdf")
    else:
        st.warning("Please load datasets first.")
