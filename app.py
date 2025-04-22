import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_clean_csv
from generate_report import generate_insight_report  # ✅ updated for reportlab

# ========== CONFIG ==========
st.set_page_config(page_title="Wine Quality Forecast", layout="wide")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCHEMA_PATH = BASE_DIR / "schema.json"
METRICS_PATH = BASE_DIR / "metrics.json"

# ========== LOAD MODEL + METADATA ==========
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "model.pkl")

@st.cache_data
def load_schema():
    with open(SCHEMA_PATH) as f: return json.load(f)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f: return json.load(f)

model = load_model()
schema = load_schema()
metrics = load_metrics()

# ========== STYLES ==========
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    </style>
""", unsafe_allow_html=True)

# ========== NAVIGATION ==========
page = st.sidebar.radio("Navigation", [
    "🏠 Overview", "�� Explore Data", "📈 Train New Model",
    "🔍 Predict Sample", "📊 Advanced Analytics", "📄 Reports", "⬇️ Export"
])

def prettify(name):
    return name.replace("_", " ").title()

# ========== PAGE: OVERVIEW ==========
if page == "🏠 Overview":
    st.title("Wine Quality Forecasting Dashboard")
    st.subheader("🔎 Model Performance")
    st.metric("R²", f"{metrics['r2']:.3f}")
    st.metric("RMSE", f"{metrics['rmse']:.3f}")
    st.metric("MAE", f"{metrics['mae']:.3f}")
    st.info(f"Trained with {metrics.get('algorithm', 'N/A')}")

# ========== PAGE: EXPLORE ==========
elif page == "📂 Explore Data":
    st.title("📂 Explore Wine Region Datasets")
    files = list(DATA_DIR.glob("*.csv"))
    selected = st.selectbox("Choose a dataset", [f.name for f in files])
    if selected:
        df = load_and_clean_csv(DATA_DIR / selected)
        st.dataframe(df.head())
        st.write("📊 Summary Statistics")
        st.dataframe(df.describe())

# ========== PAGE: TRAIN ==========
elif page == "📈 Train New Model":
    st.title("📈 Train New Model from Dataset")
    train_file = st.file_uploader("Upload training data", type="csv")

    if train_file:
        df = load_and_clean_csv(train_file)
        st.write("Dataset preview", df.head())
        target = "wine_quality_score"
        X = df[schema["features"]]
        y = df[target]

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"model_{timestamp}.pkl"
            joblib.dump(model, MODELS_DIR / version_name)
            joblib.dump(model, MODELS_DIR / "model.pkl")

            new_metrics = {
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "algorithm": "XGBoost",
                "trained_on": timestamp
            }
            with open(METRICS_PATH, "w") as f: json.dump(new_metrics, f, indent=2)

            st.success(f"✅ Model trained and saved as {version_name}")

# ========== PAGE: PREDICT ==========
elif page == "🔍 Predict Sample":
    st.title("🔍 Predict Wine Quality")
    inputs = {}
    cols = st.columns(3)
    for i, feat in enumerate(schema["features"]):
        inputs[feat] = cols[i % 3].number_input(prettify(feat), value=0.0)
    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Wine Quality Score: **{round(pred, 2)}**")

# ========== PAGE: ANALYTICS ==========
elif page == "📊 Advanced Analytics":
    st.title("📊 Advanced KPI Insights")
    st.markdown("Correlations with target: `wine_quality_score`")

    files = list(DATA_DIR.glob("*.csv"))
    selected_file = st.selectbox("Choose dataset", [f.name for f in files])
    if selected_file:
        df = load_and_clean_csv(DATA_DIR / selected_file)
        target = "wine_quality_score"
        if target not in df.columns:
            st.error(f"Target '{target}' not in dataset.")
        else:
            corr = df.corr(numeric_only=True)[target].drop(target).sort_values(key=abs, ascending=False).head(10)
            st.dataframe(corr.to_frame(name="Correlation with Wine Quality").style.background_gradient(cmap="coolwarm"))

            st.subheader("📌 KPI vs Target")
            top_feat = st.selectbox("Choose KPI", corr.index.tolist())
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=top_feat, y=target, ax=ax)
            sns.regplot(data=df, x=top_feat, y=target, ax=ax, scatter=False, color='red')
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.boxplot(data=df, x=top_feat, ax=ax2)
            st.pyplot(fig2)

# ========== PAGE: REPORTS ==========
elif page == "📄 Reports":
    st.title("📄 Generate Insight PDF Report")

    files = list(DATA_DIR.glob("*.csv"))
    selected_file = st.selectbox("Dataset to summarize", [f.name for f in files])
    if selected_file:
        df = load_and_clean_csv(DATA_DIR / selected_file)
        target = "wine_quality_score"
        if target in df.columns:
            corr_df = df.corr(numeric_only=True)[target].drop(target).to_frame(name="Correlation")

            st.subheader("📌 Choose Correlation Variable")
            top_feat = st.selectbox("Select Feature", corr_df.index.tolist())

            # Create plots
            scatter_fig, ax1 = plt.subplots()
            sns.scatterplot(data=df, x=top_feat, y=target, ax=ax1)
            sns.regplot(data=df, x=top_feat, y=target, ax=ax1, scatter=False, color='red')

            boxplot_fig, ax2 = plt.subplots()
            sns.boxplot(data=df, x=top_feat, ax=ax2)

            if st.button("Generate PDF Report"):
                pdf_path = generate_insight_report(
                    regions="Ioannina, Ionian Islands",
                    date_range="2024–2028",
                    correlation_df=corr_df,
                    scatter_fig=scatter_fig,
                    boxplot_fig=boxplot_fig
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Report", f, file_name="wine_insight_report.pdf")

# ========== PAGE: EXPORT ==========
elif page == "⬇️ Export":
    st.title("⬇️ Export Tools")
    if st.button("Download Model"):
        with open(MODELS_DIR / "model.pkl", "rb") as f:
            st.download_button("Download Model File", f.read(), "model.pkl")
    with open(SCHEMA_PATH) as f:
        st.download_button("Download Schema", f.read(), "schema.json")

    metrics_json = json.dumps(metrics, indent=2)
    st.download_button("Download Metrics", metrics_json, "metrics.json", mime="application/json")

# ========== FOOTER ==========
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Made by Baltzakis Themistoklis")
