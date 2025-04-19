import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import subprocess
import matplotlib.pyplot as plt
from scripts.utils.utils import generate_pdf_from_dict

# === Config ===
MODELS_DIR = "models"
LOGS_DIR = "logs"
DATA_DIR = "data/processed_filled"
BATCH_OUTPUT_DIR = "batch_predictions"
TARGET_COL = "wine_quality_score"
LOG_FILE_PATH = "pipeline.log"

st.set_page_config(page_title="Wine Quality Dashboard", layout="wide")
st.title("Greek Wine Quality Streamlit Dashboard")

# === Session Logs ===
if "logs" not in st.session_state:
    st.session_state.logs = []

def st_log(msg, level="INFO"):
    log_entry = f"[{level}] {msg}"
    st.session_state.logs.append(log_entry)
    st.write(log_entry)

# === State Initialization ===
if "selected_task" not in st.session_state:
    st.session_state.selected_task = "regression"

if "selected_model_path" not in st.session_state:
    def get_best_model(task):
        leaderboard_path = os.path.join(LOGS_DIR, f"leaderboard_{task}.csv")
        if os.path.exists(leaderboard_path):
            df = pd.read_csv(leaderboard_path)
            sort_col = "r2" if task == "regression" else "accuracy"
            if sort_col in df.columns and not df.empty:
                best_row = df.sort_values(by=sort_col, ascending=False).iloc[0]
                return best_row["model_path"], best_row["region"]
        return None, None
    model_path, best_region = get_best_model(st.session_state.selected_task)
    st.session_state.selected_model_path = model_path
    st.session_state.selected_region = best_region

# === Utilities ===
@st.cache_resource
def load_model(path):
    if path and os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_leaderboard(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# === Sidebar Logs ===
st.sidebar.header("Pipeline Logs")
if st.sidebar.button("Clear Logs"):
    st.session_state.logs.clear()

if os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "r") as f:
        sidebar_log = f.read()
    st.sidebar.text_area("Last Run Log File", sidebar_log, height=300)
else:
    st.sidebar.info("No pipeline log file found.")

if st.sidebar.button("Refresh Logs"):
    st.experimental_rerun()

# === Leaderboard Tab ===
def render_leaderboard_tab():
    st.header("AutoML Leaderboard")
    task_type = st.selectbox("Select Task Type", ["regression", "classification"],
                             index=["regression", "classification"].index(st.session_state.selected_task))
    st.session_state.selected_task = task_type

    leaderboard_path = os.path.join(LOGS_DIR, f"leaderboard_{task_type}.csv")
    df = load_leaderboard(leaderboard_path)

    if df is not None:
        metric = "r2" if task_type == "regression" else "accuracy"
        df = df.sort_values(by=metric, ascending=False)
        st.dataframe(df)

        selected_region = st.selectbox("Choose Region", df['region'].unique())
        region_row = df[df['region'] == selected_region]
        if not region_row.empty:
            model_path = region_row["model_path"].values[0]
        else:
            model_path = st.session_state.selected_model_path

        st.session_state.selected_model_path = model_path
        st.session_state.selected_region = selected_region

        model = load_model(model_path)
        if model:
            st.subheader(f"Model Pipeline for {selected_region}")
            st.code(str(model), language="python")
        else:
            st.warning("Model file not found or failed to load.")
    else:
        st.warning(f"No leaderboard found for task: {task_type}")

    if st.button("Run Full AutoML Pipeline"):
        with st.spinner("Training models..."):
            st_log(f"Running pipeline for task: {task_type}")
            process = subprocess.Popen(
                ["python3", "scripts/run_pipeline.py", "--task", task_type, "--full"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
            )
            for line in process.stdout:
                st_log(line.strip())
            process.wait()
            st.success("Pipeline complete. Reload to see updates.")
            st.experimental_rerun()

# === Climate Simulation Tab ===
def render_climate_simulation_tab():
    st.header("Climate Simulation")
    mt = st.slider("Mean Temperature (°C)", 20.0, 26.0, 22.5, 0.1)
    mrh = st.slider("Relative Humidity (%)", 50.0, 75.0, 60.0, 0.5)

    def predict_tas(mt, mrh): return 15.27 - 0.2 * (mt - 24.6)**2 - 0.1 * (mrh - 71.18)**2
    def predict_rs(mt, mrh): return 3.6 - 0.25 * (mt - 24.6)**2 - 0.1 * (mrh - 56.31)**2
    def predict_ts(mt, mrh): return 208.0 - 2.5 * (mt - 21.2)**2 - 1.5 * (mrh - 56.31)**2

    tas, rs, ts = predict_tas(mt, mrh), predict_rs(mt, mrh), predict_ts(mt, mrh)

    st.metric("Alcohol Strength", f"{tas:.2f} %")
    st.metric("Residual Sugars", f"{rs:.2f} g/L")
    st.metric("Total Sulfite", f"{ts:.0f} mg/L")

    param_choice = st.selectbox("Parameter to visualize", ["TAS", "Residual Sugars", "Total Sulfite"])
    func, label = {
        "TAS": (predict_tas, "TAS (%)"),
        "Residual Sugars": (predict_rs, "RS (g/L)"),
        "Total Sulfite": (predict_ts, "TS (mg/L)")
    }[param_choice]

    T, H = np.meshgrid(np.linspace(20, 26, 60), np.linspace(50, 75, 60))
    Z = func(T, H)

    fig, ax = plt.subplots()
    contour = ax.contourf(T, H, Z, cmap="YlGnBu")
    plt.colorbar(contour, ax=ax).set_label(label)
    ax.set_title(f"{param_choice} vs Climate")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    st.pyplot(fig)

    df = pd.DataFrame({
        "Temperature": [mt],
        "Humidity": [mrh],
        "TAS": [tas],
        "RS": [rs],
        "TS": [ts]
    })

    st.download_button("Download CSV", df.to_csv(index=False), file_name="climate_sim.csv")

    if st.button("Generate PDF Report"):
        pdf = generate_pdf_from_dict("Climate Simulation Report", df.iloc[0].to_dict())
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name="climate_report.pdf")

# === Prediction Tab ===
def render_prediction_tab():
    st.header("Wine Quality Prediction")

    model_path = st.session_state.get("selected_model_path")
    model = load_model(model_path)

    if not model:
        st.warning("No model loaded for prediction.")
        return

    region = st.session_state.get("selected_region", "unknown")
    data_path = os.path.join(DATA_DIR, f"combined_{region}.csv")

    if not os.path.exists(data_path):
        st.warning("Data for this region not found.")
        return

    df = pd.read_csv(data_path)
    features = df.select_dtypes(include=[float, int]).drop(columns=[TARGET_COL], errors="ignore")

    st.subheader("Predict from Row")
    row_index = st.slider("Choose Row", 0, len(features) - 1)
    row = features.iloc[row_index]
    st.write(row)
    pred = model.predict(row.values.reshape(1, -1))[0]
    st.success(f"Prediction: {pred:.2f}")

    st.subheader("Manual Prediction")
    inputs = {
        col: st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        for col in features.columns
    }

    if st.button("Predict from Manual Input"):
        input_df = pd.DataFrame([inputs])
        result = model.predict(input_df)[0]
        st.success(f"Predicted Quality: {result:.2f}")
        input_df["Predicted Quality"] = result
        st.download_button("Download CSV", input_df.to_csv(index=False), file_name="manual_prediction.csv")

        if st.button("Generate PDF"):
            pdf = generate_pdf_from_dict("Manual Prediction Report", input_df.iloc[0].to_dict())
            with open(pdf, "rb") as f:
                st.download_button("Download PDF", f, file_name="manual_prediction.pdf")

# === Reports Tab ===
def render_reports_tab():
    st.header("Reports")
    for task in ["regression", "classification"]:
        leaderboard_path = os.path.join(LOGS_DIR, f"leaderboard_{task}.csv")
        df = load_leaderboard(leaderboard_path)
        if df is not None:
            st.subheader(f"{task.capitalize()} Leaderboard")
            st.dataframe(df)
            st.download_button(f"Download {task} leaderboard", df.to_csv(index=False), file_name=f"leaderboard_{task}.csv")

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["Leaderboard", "Climate Simulation", "Prediction", "Reports"])

with tab1: render_leaderboard_tab()
with tab2: render_climate_simulation_tab()
with tab3: render_prediction_tab()
with tab4: render_reports_tab()

# === Debug Logs ===
with st.expander("Debug Logs"):
    for line in st.session_state.logs[-100:]:
        st.text(line)
