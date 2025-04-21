import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from PIL import Image

# === PATHS ===
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# === LOAD MODEL & METADATA ===
model = joblib.load(MODELS_DIR / "model.pkl")

with open(BASE_DIR / "metrics.json") as f:
    metrics = json.load(f)

with open(BASE_DIR / "schema.json") as f:
    schema = json.load(f)

with open(DATA_DIR / "climate_lookup.json") as f:
    climate_data = json.load(f)

# === PAGE CONFIG ===
st.set_page_config(page_title="🍷 Wine Quality & Climate Forecast", layout="wide")
st.title("🍷 Wine Quality & Climate Forecast Dashboard")

# === NAVIGATION ===
section = st.sidebar.radio("Navigation", [
    "Overview", "Wine Quality Explorer", "Climate Effects Simulator",
    "Single Prediction", "Batch Predictions", "Forecast Performance", "Download Center"
])

# === 1. OVERVIEW ===
if section == "Overview":
    st.subheader("📌 Project Summary")
    st.markdown("""
    Predict wine quality using climate and viticultural features.
    Visualize region datasets, feature impact, and forecast results.
    """)
    st.json(metrics)

# === 2. WINE QUALITY EXPLORER ===
elif section == "Wine Quality Explorer":
    st.subheader("🍇 Explore Wine Datasets")

    dataset_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith("climate")])
    if dataset_files:
        selected_file = st.selectbox("Select a dataset", dataset_files)
        df = pd.read_csv(DATA_DIR / selected_file)
        st.write(f"### Preview: {selected_file}")
        st.dataframe(df.head())
        st.write("### Summary Stats")
        st.dataframe(df.describe())
    else:
        st.warning("No datasets found in /data.")

# === 3. CLIMATE EFFECTS SIMULATOR ===
elif section == "Climate Effects Simulator":
    st.subheader("🌤️ Climate Impact Simulation")
    region = st.selectbox("Choose a wine region", sorted(climate_data.keys()))
    region_climate = climate_data[region]

    temp = st.slider("Mean Temperature (°C)", 15.0, 30.0, region_climate["mean_temp"])
    humidity = st.slider("Relative Humidity (%)", 40.0, 90.0, region_climate["humidity"])

    st.markdown("#### 🔎 Interpretation")
    if temp > 24 and humidity < 60:
        st.success("↑ Higher alcohol and sugar levels likely.")
    elif temp < 21 and humidity > 70:
        st.warning("↓ Lower sulfite & alcohol levels expected.")
    else:
        st.info("No strong shift expected.")

# === 4. SINGLE PREDICTION (FULL FEATURES) ===
elif section == "Single Prediction":
    st.subheader("🔮 Predict Wine Quality (Manual Input)")
    st.markdown("Fill in values for each input feature:")

    input_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(schema["features"]):
        col = cols[i % 3]
        input_values[feat] = col.number_input(feat, value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_values])
        pred = model.predict(input_df)[0]
        st.success(f"📈 Predicted Wine Quality: **{round(pred, 2)}**")

# === 5. BATCH PREDICTIONS ===
elif section == "Batch Predictions":
    st.subheader("📥 Predict from Uploaded CSV")
    st.markdown("Upload a CSV file with the **same columns as `schema.json` features**.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Validate
        expected_cols = schema["features"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(df)
            df["Predicted_Quality"] = preds
            st.success("✅ Predictions completed")
            st.dataframe(df.head())

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", data=csv_data, file_name="wine_predictions.csv", mime="text/csv")

# === 6. FORECAST PERFORMANCE ===
elif section == "Forecast Performance":
    st.subheader("📊 Visual Model Evaluation")
    st.image(ASSETS_DIR / "predictions_vs_actuals.png", use_column_width=True)
    st.image(ASSETS_DIR / "feature_importance.png", use_column_width=True)

# === 7. DOWNLOADS ===
elif section == "Download Center":
    st.subheader("⬇️ Export Samples")

    st.markdown("Download an example input file with the correct format:")
    example_df = pd.DataFrame([{
        feat: 0.0 for feat in schema["features"]
    }])
    csv = example_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Template", csv, file_name="wine_input_template.csv", mime="text/csv")

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.caption("Built with 🍇 and Streamlit")
