import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_clean_csv
from generate_report import generate_insight_report

# ========== CONFIG ==========
st.set_page_config(page_title="Wine Quality Forecast", layout="wide")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCHEMA_PATH = BASE_DIR / "schema.json"
METRICS_PATH = BASE_DIR / "metrics.json"

# ========== REGION PROFILES ==========
region_profiles = {
    "Amyntaio": {
        "description": "Northern Greece's highest-altitude wine region, known for Xinomavro wines.",
        "climate": "Continental",
        "varietal": "Xinomavro",
        "emoji": "🏔️"
    },
    "Rapsani": {
        "description": "Nestled on Mount Olympus, famous for full-bodied red blends.",
        "climate": "Mountainous Mediterranean",
        "varietal": "Xinomavro, Krasato, Stavroto",
        "emoji": "⛰️"
    },
    "Santorini": {
        "description": "Volcanic island producing Assyrtiko with unique minerality.",
        "climate": "Island",
        "varietal": "Assyrtiko",
        "emoji": "🌋"
    },
    "Mantineia": {
        "description": "Peloponnese plateau with cool microclimate for Moschofilero whites.",
        "climate": "Continental",
        "varietal": "Moschofilero",
        "emoji": "��️"
    },
    "Nemea": {
        "description": "Largest red-wine appellation, producing rich Agiorgitiko.",
        "climate": "Mediterranean",
        "varietal": "Agiorgitiko",
        "emoji": "��"
    },
    "Naoussa": {
        "description": "Home of structured Xinomavro with long aging potential.",
        "climate": "Continental",
        "varietal": "Xinomavro",
        "emoji": "🧱"
    },
    "Patras": {
        "description": "Diverse region near the Gulf, known for sweet and dry styles.",
        "climate": "Mediterranean Coastal",
        "varietal": "Mavrodaphne, Roditis",
        "emoji": "🌊"
    }
}

def get_region_file_map():
    files = list(DATA_DIR.glob("*.csv"))
    def extract_region_name(path):
        name = path.stem.lower()
        for word in ["combined", "filled", "-", "_"]:
            name = name.replace(word, "")
        return name.strip().title()
    return {extract_region_name(f): f for f in files}

# ========== LOAD ==========
@st.cache_resource
def load_model(): return joblib.load(MODELS_DIR / "model.pkl")

@st.cache_data
def load_schema():
    with open(SCHEMA_PATH) as f: return json.load(f)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f: return json.load(f)

model = load_model()
schema = load_schema()
metrics = load_metrics()

def prettify(name): return name.replace("_", " ").title()

# ========== NAV ==========
page = st.sidebar.radio("Navigation", [
    "🏠 Overview", "📂 Explore Data", "📈 Train New Model",
    "🔍 Predict Sample", "📊 Advanced Analytics", "�� Reports", "⬇️ Export"
])

# ========== PAGE: OVERVIEW ==========
if page == "🏠 Overview":
    st.title("Wine Quality Forecasting Dashboard")
    st.metric("R²", f"{metrics['r2']:.3f}")
    st.metric("RMSE", f"{metrics['rmse']:.3f}")
    st.metric("MAE", f"{metrics['mae']:.3f}")
    st.info(f"Trained with {metrics.get('algorithm', 'N/A')}")

# ========== PAGE: EXPLORE ==========
elif page == "📂 Explore Data":
    st.title("📂 Explore Wine Region Datasets")
    region_map = get_region_file_map()

    selected_region = st.selectbox("Choose a region", list(region_map.keys()))
    if selected_region:
        df = load_and_clean_csv(region_map[selected_region])
        profile = region_profiles.get(selected_region, {})

        st.markdown(f"""
        ### {profile.get('emoji', '')} {selected_region}
        **Climate:** {profile.get('climate', 'N/A')}  
        **Varietals:** *{profile.get('varietal', 'N/A')}*

        {profile.get('description', '')}
        """)
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
    st.title("�� Predict Wine Quality")
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
    region_map = get_region_file_map()

    selected_region = st.selectbox("Choose region for analysis", list(region_map.keys()))
    if selected_region:
        df = load_and_clean_csv(region_map[selected_region])
        target = "wine_quality_score"
        if target not in df.columns:
            st.error(f"Target '{target}' not in dataset.")
        else:
            corr_df = df.corr(numeric_only=True)[target].drop(target).to_frame(name="Correlation")
            st.dataframe(corr_df.sort_values(by="Correlation", key=abs, ascending=False).head(15))

            st.subheader("📌 Visualize Top Feature")
            top_feat = st.selectbox("Select a KPI", corr_df.dropna().sort_values(by="Correlation", key=abs, ascending=False).index.tolist())

            fig1, ax1 = plt.subplots()
            sns.scatterplot(data=df, x=top_feat, y=target, ax=ax1)
            sns.regplot(data=df, x=top_feat, y=target, ax=ax1, scatter=False, color='red')
            ax1.set_title(f"{top_feat} vs Wine Quality")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.boxplot(data=df, x=top_feat, ax=ax2)
            ax2.set_title(f"Distribution of {top_feat}")
            st.pyplot(fig2)

# ========== PAGE: REPORTS ==========
elif page == "📄 Reports":
    st.title("📄 Generate Scientific PDF Report")
    region_map = get_region_file_map()

    selected_region = st.selectbox("Choose region to analyze", list(region_map.keys()))
    if selected_region:
        df = load_and_clean_csv(region_map[selected_region])
        profile = region_profiles.get(selected_region, {})
        target = "wine_quality_score"

        if target in df.columns:
            corr_df = df.corr(numeric_only=True)[target].drop(target).to_frame(name="Correlation")
            top_feat = st.selectbox("Choose Top KPI to Plot", corr_df.dropna().sort_values(by="Correlation", key=abs, ascending=False).index.tolist())

            scatter_fig, ax1 = plt.subplots()
            sns.scatterplot(data=df, x=top_feat, y=target, ax=ax1)
            sns.regplot(data=df, x=top_feat, y=target, ax=ax1, scatter=False, color='red')

            boxplot_fig, ax2 = plt.subplots()
            sns.boxplot(data=df, x=top_feat, ax=ax2)

            include_appendix = st.checkbox("Include full correlation matrix (Appendix)", value=False)
            dashboard_url = st.text_input("Streamlit App URL for QR Code", value="https://your-streamlit-app")

            region_description = f"{selected_region} — {profile.get('climate', 'N/A')} region, notable for {profile.get('varietal', 'N/A')}. {profile.get('description', '')}"

            if st.button("Generate PDF Report"):
                pdf_path = generate_insight_report(
                    regions=region_description,
                    date_range="2024–2028",
                    correlation_df=corr_df,
                    scatter_fig=scatter_fig,
                    boxplot_fig=boxplot_fig,
                    metrics=metrics,
                    include_appendix=include_appendix,
                    dashboard_url=dashboard_url
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
    with open(METRICS_PATH) as f:
        st.download_button("Download Metrics", f.read(), "metrics.json", mime="application/json")

# ========== FOOTER ==========
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍🔬 Built by Baltzakis Themistoklis")
