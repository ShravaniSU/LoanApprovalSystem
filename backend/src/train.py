import pandas as pd
import joblib
import argparse
import mltable
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Load dataset
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to Training dataset")
parser.add_argument("--model_output", type=str, required=False, help="Path to save the trained model")

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

# Train model block with MLflow tracking
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

    # Evaluate model and calculate comprehensive metrics
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    
    # Log metrics manually to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    print(f"Model Metrics -> Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Local fallback save just in case
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Conditional Quality Gate for Model Registration
    # Gates: Accuracy > 70% (0.70) AND Precision/Recall/F1 > 80% (0.80)
    if acc > 0.70 and precision > 0.80 and recall > 0.80 and f1 > 0.80:
        print("Performance gates passed! Registering model to Azure ML...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="loan_prediction_model",
            registered_model_name="loan_prediction_model"
        )
        print("Model successfully registered in Azure ML Model Registry!")
    else:
        print("Model did not meet all performance criteria thresholds. Skipping model registration.")