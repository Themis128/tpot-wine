import streamlit as st
import pandas as pd
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import locale
import os
import subprocess
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# --- EU number formatting ---
try:
    locale.setlocale(locale.LC_NUMERIC, "en_IE.UTF-8")
except:
    locale.setlocale(locale.LC_NUMERIC, "")

# --- Paths ---
CLEAN_DATA_PATH = "data/cleaned_combined_data.csv"
MODEL_PATH = "models/version_latest/model.pkl"
FEATURE_NAMES_PATH = "models/version_latest/feature_names.txt"
METADATA_PATH = "models/version_latest/model_timestamp.txt"
ASSETS_PATH = "assets"
RETRAIN_SCRIPT = "scripts/refit_model_with_readable_names.py"

# --- Constants ---
TARGET = "Wine Quality"
MIN_FEATURE_MATCH = 0.7

RENAME_MAP = {
    'temperature_2m_mean': 'Avg Temp (2m)',
    'precipitation_sum': 'Total Precipitation',
    'et0_fao_evapotranspiration_x': 'ET0 FAO (X)',
    'et0_fao_evapotranspiration_y': 'ET0 FAO (Y)',
    'wine_quality_score': 'Wine Quality',
    'region': 'Region',
    'date': 'Date'
}

# --- Utility ---
def get_model_timestamp():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return f.read().strip()
    return "Unknown"

@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        features = [line.strip() for line in f]
    return model, features

@st.cache_data
def load_cleaned_data():
    return pd.read_csv(CLEAN_DATA_PATH)

def safe_predict(df, model, feature_cols):
    available = [col for col in feature_cols if col in df.columns]
    missing = list(set(feature_cols) - set(available))
    match_ratio = len(available) / len(feature_cols)

    with st.expander("🔍 Feature availability debug info"):
        st.json({
            "Expected features": len(feature_cols),
            "Available": len(available),
            "Missing": missing
        })

    if match_ratio < MIN_FEATURE_MATCH:
        st.warning("⚠️ Too few matching features. Auto-retraining model...")

        with st.spinner("🔄 Retraining model..."):
            result = subprocess.run(["python3", RETRAIN_SCRIPT], capture_output=True, text=True)

        if result.returncode != 0:
            st.error("❌ Retraining failed.")
            st.text(result.stdout + result.stderr)
            return None

        st.success("✅ Model retrained successfully.")

        new_model, new_features = load_model_and_features()
        return safe_predict(df, new_model, new_features)

    for col in available:
        df[col] = df[col].fillna(df[col].mean())

    return model.predict(df[available])

def plot_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importance.sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(8, top_n * 0.35))
        sns.barplot(x=top_features.values, y=top_features.index, ax=ax, color="skyblue")
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance (0–1)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.2f}".replace(".", ",")))
        st.pyplot(fig)

def evaluate_model(model, X, y):
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    euro = lambda x: f"{x:,.3f}".replace(".", ",")

    st.markdown(f"""
    - **R² Score**: `{euro(r2)}`
    - **Mean Absolute Error (MAE)**: `{euro(mae)}`
    - **Root Mean Squared Error (RMSE)**: `{euro(rmse)}`
    """)

def show_shap_summary(model, X, feature_names):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader("🔎 SHAP Global Explanation")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    st.pyplot(fig)

# --- Streamlit Layout ---
st.set_page_config("🍇 Wine Quality Dashboard", layout="wide")
st.sidebar.title("🍷 Wine Quality Explorer")
st.sidebar.markdown(f"🕒 **Last Trained:** `{get_model_timestamp()}`")
st.title("🍇 Climate & Wine Quality")

# --- Manual Retrain Button ---
if st.sidebar.button("🔁 Retrain Model Now"):
    with st.spinner("Running retraining script..."):
        result = subprocess.run(["python3", RETRAIN_SCRIPT], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Model retrained successfully.")
        else:
            st.error("❌ Retraining failed.")
            st.text(result.stdout + result.stderr)

# --- Load everything ---
df = load_cleaned_data()
model, feature_cols = load_model_and_features()
df = df.rename(columns=RENAME_MAP)
df = df.loc[:, ~df.columns.duplicated()]
display_df = df.copy()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Visuals",
    "🤖 Model Predictions",
    "📄 Raw Data",
    "🧠 Your CSV",
    "📷 Model Metrics"
])

with tab1:
    st.subheader("📊 Climate Variables vs Wine Quality")
    plot_options = [col for col in display_df.select_dtypes(include='number').columns if col != TARGET]
    if plot_options:
        xcol = st.selectbox("Choose a climate variable", plot_options)
        st.vega_lite_chart(display_df, {
            "mark": "point",
            "encoding": {
                "x": {"field": xcol, "type": "quantitative"},
                "y": {"field": TARGET, "type": "quantitative"},
                "color": {"field": "Region", "type": "nominal"} if "Region" in display_df.columns else {}
            }
        }, use_container_width=True)

with tab2:
    st.subheader("🤖 Predictions on Full Dataset")
    preds = safe_predict(df.copy(), model, feature_cols)
    if preds is not None:
        df = df.copy()
        df["Predicted Quality"] = preds
        display_df = df.copy()

        st.success("✅ Predictions completed.")
        st.dataframe(display_df[["Date", "Region", "Wine Quality", "Predicted Quality"]].dropna(), use_container_width=True)

        with st.expander("📥 Download Predictions"):
            st.download_button("Download CSV", df.to_csv(index=False), file_name="predictions.csv")

        with st.expander("📊 Feature Importance"):
            plot_feature_importance(model, feature_cols)

        with st.expander("🧠 SHAP Explainability"):
            show_shap_summary(model, df[feature_cols], feature_cols)

        with st.expander("🧪 Evaluation Metrics"):
            valid_rows = df.dropna(subset=["Predicted Quality"])
            evaluate_model(model, valid_rows[feature_cols], valid_rows[TARGET])

with tab3:
    st.subheader("📄 Raw Data Sample")
    st.dataframe(display_df.head(100), use_container_width=True)

with tab4:
    st.subheader("🧠 Predict from Your CSV")
    uploaded = st.file_uploader("📥 Upload a CSV file", type="csv")
    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            user_df = user_df.loc[:, ~user_df.columns.str.contains("^Unnamed")]
            user_df = user_df.dropna(axis=1, how="all")
            user_df = user_df.rename(columns=RENAME_MAP)
            user_df = user_df.loc[:, ~user_df.columns.duplicated()]

            available = [col for col in feature_cols if col in user_df.columns]
            for col in available:
                user_df[col] = user_df[col].fillna(user_df[col].mean())

            if len(available) < len(feature_cols) * MIN_FEATURE_MATCH:
                st.warning("⚠️ Not enough features. Auto-retraining will be triggered.")
                preds = safe_predict(user_df.copy(), model, feature_cols)
            else:
                preds = model.predict(user_df[available])

            if preds is not None:
                user_df = user_df.copy()
                user_df["Predicted Quality"] = preds
                st.success("✅ Prediction complete.")
                st.dataframe(user_df[["Predicted Quality"] + available].head())

                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
                st.download_button("📤 Download Results", user_df.to_csv(index=False), file_name=f"user_predictions_{timestamp}.csv")

                with st.expander("🧠 SHAP Explanation (first 5 rows)"):
                    explainer = shap.Explainer(model, user_df[available])
                    shap_values = explainer(user_df[available])
                    for i in range(min(5, len(user_df))):
                        st.markdown(f"#### Row {i+1}")
                        fig = shap.plots.waterfall(shap_values[i], show=False)
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")

with tab5:
    st.subheader("📷 Precomputed Model Visuals")
    st.markdown("These visuals were generated during model training.")

    image_paths = {
        "Normalized Confusion Matrix": "assets/confusion_matrix_normalized.png",
        "Confusion Matrix": "assets/confusion_matrix.png",
        "ROC Curve": "assets/roc_curve.png",
        "Precision-Recall Curve": "assets/precision_recall_curve.png",
        "Learning Curve": "assets/learning_curves.png"
    }

    for title, path in image_paths.items():
        if os.path.exists(path):
            with st.expander(title):
                st.image(path, use_container_width=True)
        else:
            st.warning(f"⚠️ File not found: {path}")
