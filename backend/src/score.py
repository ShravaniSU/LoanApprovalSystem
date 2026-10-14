import os
import logging
import json
import mlflow.pyfunc
import pandas as pd

def init():
    global model
    try:
        model_dir = os.getenv("AZUREML_MODEL_DIR")
        logging.info(f"Model directory path: {model_dir}")
        
        model_path = model_dir
        for root, dirs, files in os.walk(model_dir):
            if "MLmodel" in files:
                model_path = root
                break
                
        logging.info(f"Loading MLflow model from: {model_path}")
        model = mlflow.pyfunc.load_model(model_path)
        logging.info("Model initialized and loaded successfully!")
        
    except Exception as e:
        logging.error(f"Initialization failed: {str(e)}")
        raise e

def run(raw_data):
    try:
        logging.info("Request received.")
        body = json.loads(raw_data)
        
        if isinstance(body, dict):
            if "input_data" in body:
                payload = body["input_data"]
            elif "data" in body:
                payload = body["data"]
            else:
                payload = body
        else:
            payload = body

        if isinstance(payload, list):
            input_df = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            input_df = pd.DataFrame([payload])
        else:
            raise ValueError("Unsupported JSON payload structure.")

        # --- EXACT TYPE & ENCODING MAPPINGS TO MATCH MODEL SCHEMA ---
        
        # Gender: Male -> 1, Female -> 0
        if "Gender" in input_df.columns:
            if input_df["Gender"].dtype == object:
                input_df["Gender"] = input_df["Gender"].map({"Male": 1, "Female": 0})
            input_df["Gender"] = input_df["Gender"].astype(int)

        # Married: boolean
        if "Married" in input_df.columns:
            if input_df["Married"].dtype == object:
                input_df["Married"] = input_df["Married"].map({"Yes": True, "No": False, "true": True, "false": False})
            input_df["Married"] = input_df["Married"].astype(bool)

        # Dependents: long/int
        if "Dependents" in input_df.columns:
            input_df["Dependents"] = input_df["Dependents"].astype(str).str.replace('+', '', regex=False)
            input_df["Dependents"] = pd.to_numeric(input_df["Dependents"], errors='coerce').fillna(0).astype(int)

        # Education: long/int
        if "Education" in input_df.columns:
            if input_df["Education"].dtype == object:
                input_df["Education"] = input_df["Education"].map({"Graduate": 1, "Not Graduate": 0})
            input_df["Education"] = input_df["Education"].astype(int)

        # Self_Employed: boolean
        if "Self_Employed" in input_df.columns:
            if input_df["Self_Employed"].dtype == object:
                input_df["Self_Employed"] = input_df["Self_Employed"].map({"Yes": True, "No": False, "true": True, "false": False})
            input_df["Self_Employed"] = input_df["Self_Employed"].astype(bool)

        # ApplicantIncome: long/int
        if "ApplicantIncome" in input_df.columns:
            input_df["ApplicantIncome"] = input_df["ApplicantIncome"].astype(int)

        # CoapplicantIncome: long/int (matches schema error)
        if "CoapplicantIncome" in input_df.columns:
            input_df["CoapplicantIncome"] = input_df["CoapplicantIncome"].astype(int)

        # LoanAmount: double/float
        if "LoanAmount" in input_df.columns:
            input_df["LoanAmount"] = input_df["LoanAmount"].astype(float)

        # Loan_Amount_Term: double/float
        if "Loan_Amount_Term" in input_df.columns:
            input_df["Loan_Amount_Term"] = input_df["Loan_Amount_Term"].astype(float)

        # Credit_History: boolean
        if "Credit_History" in input_df.columns:
            input_df["Credit_History"] = input_df["Credit_History"].astype(bool)

        # Property_Area: long/int
        if "Property_Area" in input_df.columns:
            if input_df["Property_Area"].dtype == object:
                input_df["Property_Area"] = input_df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})
            input_df["Property_Area"] = input_df["Property_Area"].astype(int)

        logging.info(f"Input data shape after conversion: {input_df.shape}")
        
        # Generate predictions
        predictions = model.predict(input_df)
        
        return {"predictions": predictions.tolist()}
        
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}