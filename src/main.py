# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict

# Load the preprocessor and models
preprocessor = joblib.load('preprocessor.pkl')
log_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
svm_model = joblib.load('svm_model.pkl')

app = FastAPI()

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

# Define the endpoint for CSV file uploads
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file.file)
        
        # Preprocess the input data
        X_processed = preprocess_data(df)
        
        # Get predictions from all models
        predictions = {
            "Logistic Regression": log_model.predict(X_processed).tolist(),
            "Random Forest": rf_model.predict(X_processed).tolist(),
            "XGBoost": xgb_model.predict(X_processed).tolist(),
            "SVM": svm_model.predict(X_processed).tolist()
        }
        
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to preprocess the data
def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    # Preprocess the input data
    X = df.copy()
    
    # Apply the same preprocessing pipeline used for training
    X_processed = preprocessor.transform(X)
    
    return X_processed

# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
