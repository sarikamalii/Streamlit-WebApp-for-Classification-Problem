import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)

# Step 1: Read the data
def read_data(file_name):
    file_path = f"{parent_dir}/Data/{file_name}"
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        return df

# Step 2: Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if there are only numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(numerical_cols) > 0:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Impute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scale the numerical features based on scaler_type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) > 0:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test

# Step 3: Train the model
def train_model(X_train, y_train, model, model_name):
    # training the selected model
    model.fit(X_train, y_train)
    
    # create directory if it does not exist
    model_dir = f"{parent_dir}/trained_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # saving the trained model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)
    return accuracy, precision, recall, f1
