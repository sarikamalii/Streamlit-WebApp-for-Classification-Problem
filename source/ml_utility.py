import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

# Preprocessing function with missing value and outlier handling
def preprocess_data(data, treat_outliers=True, target_column="Loan_Status"):
    st.write("### Data before Missing Value Treatment")
    st.write(data.isnull().sum())

    # Missing Value Treatment
    imputer = SimpleImputer(strategy="mean")
    for col in data.select_dtypes(include=["float64", "int64"]).columns:
        if col != target_column:
            data[col] = imputer.fit_transform(data[[col]])

    st.write("### Data after Missing Value Treatment")
    st.write(data.isnull().sum())
    
    # Outlier Treatment
    if treat_outliers:
        for col in data.select_dtypes(include=["float64", "int64"]).columns:
            if col != target_column:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower_bound, upper_bound)

    # Encoding categorical variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Separating target and features
    X = data.drop(target_column, axis=1)
    y = data[target_column].apply(lambda x: 1 if x == "Y" else 0)

    # Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Function to train and evaluate models
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }
    
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        }
    
    return results
