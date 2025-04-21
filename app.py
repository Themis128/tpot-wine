import streamlit as st
import pandas as pd
import joblib
import json
import os
from PIL import Image

# File paths
DATASETS_PATH = "/mnt/c/Users/baltz/Desktop/Models/processed_filled"
METRICS_PATH = "metrics.json"
MODEL_PATH = "model.pkl"
FEATURE_IMPORTANCE_PATH = "feature_importance.png"
PREDICTIONS_PATH = "predictions_vs_actuals.png"
SCHEMA_PATH = "schema.json"

# Load files
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

model = joblib.load(MODEL_PATH)

# Streamlit app
st.title("XGBoost Forecasting Dashboard")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Datasets", "Predictions", "Feature Importance", "Make a Prediction"])

if page == "Overview":
    st.header("Model Overview")
    st.write("### Metrics")
    st.json(metrics)

    st.write("### Schema")
    st.json(schema)

elif page == "Datasets":
    st.header("Explore Datasets")
    # List all CSV files in the datasets directory
    csv_files = [f for f in os.listdir(DATASETS_PATH) if f.endswith(".csv")]

    if not csv_files:
        st.write("No datasets found in the specified path.")
    else:
        # Allow user to select a dataset
        selected_file = st.selectbox("Select a dataset", csv_files)

        # Load and display the selected dataset
        if selected_file:
            file_path = os.path.join(DATASETS_PATH, selected_file)
            df = pd.read_csv(file_path)
            st.write(f"### {selected_file}")
            st.dataframe(df)

            # Display basic statistics
            st.write("### Dataset Statistics")
            st.write(df.describe())

elif page == "Predictions":
    st.header("Predictions vs Actuals")
    st.image(PREDICTIONS_PATH, caption="Predictions vs Actuals", use_column_width=True)

elif page == "Feature Importance":
    st.header("Feature Importance")
    st.image(FEATURE_IMPORTANCE_PATH, caption="Feature Importance", use_column_width=True)

elif page == "Make a Prediction":
    st.header("Make a Prediction")
    st.write("Enter feature values to make a prediction:")

    # Create input fields for each feature
    input_data = {}
    for feature in schema["features"]:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

    # Predict button
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.write(f"### Prediction: {prediction[0]}")

# Footer
st.sidebar.info("Streamlit Forecasting Dashboard")
