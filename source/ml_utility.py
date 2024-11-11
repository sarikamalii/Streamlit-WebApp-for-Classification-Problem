
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_data(file):
    return pd.read_csv(file)

def eda_summary(data):
    st.write("Shape of data:", data.shape)
    st.write("Data Types:\n", data.dtypes)
    st.write("Class Distribution:\n", data.iloc[:, -1].value_counts())
    st.write("Missing Values Before Treatment:")
    st.write(data.isnull().sum())

def preprocess_data(data, scaling_method="Standard", treat_outliers=False, target_column="loan_status"):
    # Copy of the original data before any changes
    data_before = data.copy()

    # Handling missing values: fill with median for numerical, mode for categorical
    for column in data.columns:
        if data[column].dtype == "object":
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    st.write("Missing Values After Treatment:")
    st.write(data.isnull().sum())
    st.write("Dataset Before Missing Value Treatment:")
    st.dataframe(data_before.head())
    st.write("Dataset After Missing Value Treatment:")
    st.dataframe(data.head())

    # Encoding categorical columns
    label_encoders = {}
    for column in data.select_dtypes(include="object").columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Outlier treatment on all features except the target column
    if treat_outliers:
        X = data.drop(columns=[target_column])
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        # Remove rows where outliers are found in any feature (excluding target column)
        X_outliers_removed = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        y_outliers_removed = data.loc[X_outliers_removed.index, target_column]
        data = X_outliers_removed.join(y_outliers_removed)
        st.write("Outliers Removed:", X.shape[0] - X_outliers_removed.shape[0], "rows removed.")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Scaling
    if scaling_method == "Standard":
        scaler = StandardScaler()
    else:
        scaler = None

    if scaler:
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
    }
    
    results = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results["Model"].append(model_name)
        results["Accuracy"].append(accuracy_score(y_test, y_pred))
        results["Precision"].append(precision_score(y_test, y_pred, average='weighted'))
        results["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
        results["F1 Score"].append(f1_score(y_test, y_pred, average='weighted'))

    return pd.DataFrame(results)

def compare_with_without_outliers(data, target_column="loan_status"):
    # Without outlier treatment
    X_no_outliers, y_no_outliers = preprocess_data(data.copy(), treat_outliers=False, target_column=target_column)
    X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(X_no_outliers, y_no_outliers, test_size=0.3, random_state=42)
    results_no_outliers = train_and_evaluate_models(X_train_no, y_train_no, X_test_no, y_test_no)
    st.write("Results without Outlier Treatment")
    st.dataframe(results_no_outliers)

    # With outlier treatment
    X_with_outliers, y_with_outliers = preprocess_data(data.copy(), treat_outliers=True, target_column=target_column)
    X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(X_with_outliers, y_with_outliers, test_size=0.3, random_state=42)
    results_with_outliers = train_and_evaluate_models(X_train_with, y_train_with, X_test_with, y_test_with)
    st.write("Results with Outlier Treatment")
    st.dataframe(results_with_outliers)
    
    return results_no_outliers, results_with_outliers
