
import streamlit as st
import pandas as pd
from ml_utils import (
    load_data,
    eda_summary,
    preprocess_data,
    train_and_evaluate_models
)

# Page setup
st.set_page_config(page_title="Classification App", layout="wide")
st.title("Classification Model Comparison")

# Sidebar for file upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Step 2: EDA
    st.sidebar.header("2. Exploratory Data Analysis")
    if st.sidebar.button("Perform EDA"):
        st.write("### EDA Summary")
        eda_summary(data)

    # Step 3: Preprocessing Options
    st.sidebar.header("3. Preprocessing")
    scaling = st.sidebar.selectbox("Scaling Method", ["None", "Standard", "MinMax"])
    processed_data = preprocess_data(data, scaling_method=scaling)

    # Step 4: Model Training and Evaluation
    st.sidebar.header("4. Model Training")
    if st.sidebar.button("Train & Evaluate Models"):
        st.write("### Model Performance Metrics")
        results = train_and_evaluate_models(processed_data)
        st.write(results)

else:
    st.info("Please upload a dataset to get started.")
