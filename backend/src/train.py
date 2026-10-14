import pandas as pd
import joblib
import argparse
import mltable
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load dataset
#df = pd.read_csv("data/loan-predictor-traindata.csv")
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to Training dataset")

args = parser.parse_args()
tbl = mltable.load(args.data_path)
df = tbl.to_pandas_dataframe()

# Drop Loan_ID if present
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
# Split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
mlflow.sklearn.autolog()

# Train model
with mlflow.start_run():
    # Define hyperparameters
    n_estimators = 100
    max_depth = 5
    
    # Log parameters manually
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Log metrics manually
    mlflow.log_metric("accuracy", acc)
    print(f"Model Accuracy: {acc}")

    # Log and save the model to MLflow artifact store
    mlflow.sklearn.log_model(model, "loan_prediction_model")
    print("Model successfully logged to MLflow!")