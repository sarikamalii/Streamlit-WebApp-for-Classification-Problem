
import streamlit as st
from ml_utility import load_data, eda_summary, compare_with_without_outliers

st.title("Classification Model Comparison with and without Outliers")
data_file = st.file_uploader("Upload Dataset", type=["csv"])

if data_file is not None:
    data = load_data(data_file)
    eda_summary(data)
    results_no_outliers, results_with_outliers = compare_with_without_outliers(data)
    
    st.write("Comparison Summary")
    st.write("Results without Outlier Treatment:")
    st.dataframe(results_no_outliers)
    
    st.write("Results with Outlier Treatment:")
    st.dataframe(results_with_outliers)
