
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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

    # Display missing values before treatment
    st.write("Missing Values Before Treatment:")
    st.write(data.isnull().sum())

def preprocess_data(data, scaling_method="None", treat_outliers=False):
    data_before = data.copy()

    # Handling missing values: fill with median for numerical, mode for categorical
    for column in data.columns:
        if data[column].dtype == "object":
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    # Display missing values after treatment
    st.write("Missing Values After Treatment:")
    st.write(data.isnull().sum())

    # Show dataset before and after missing value treatment
    st.write("Dataset Before Missing Value Treatment:")
    st.dataframe(data_before.head())
    
    st.write("Dataset After Missing Value Treatment:")
    st.dataframe(data.head())

    # Encoding categorical columns
    label_encoders = {}
    for column in data.select_dtypes(include="object").columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Outlier treatment if selected
    if treat_outliers:
        X = data.iloc[:, :-1]
        # Identifying and treating outliers using IQR (Interquartile Range)
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        X_outliers_removed = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        y_outliers_removed = data.iloc[X_outliers_removed.index, -1]
        data = X_outliers_removed.join(y_outliers_removed)
        st.write("Outliers Removed: ", X.shape[0] - X_outliers_removed.shape[0], "rows removed.")
    else:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

    # Scaling
    if scaling_method == "Standard":
        scaler = StandardScaler()
    elif scaling_method == "MinMax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

def train_and_evaluate_models(data, treat_outliers=False):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

