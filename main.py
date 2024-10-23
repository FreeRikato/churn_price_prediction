from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Global variables to store current model and dataset
current_model = None
current_dataset = None
current_dataset_columns = None

# Load your models and datasets (pre-trained models)
models = {
    "logistic_regression": joblib.load("./models/logistic_regression_model.pkl"),
    "random_forest": joblib.load("./models/random_forest_model.pkl"),
}

datasets = {
    "bank": "./data/BankChurners.csv",
    "ecommerce": "./data/E Commerce Dataset.csv",
    "internet_service": "./data/internet_service_churn.csv",
    "telecom": "./data/orange_telecom.csv",
}


class SetModelPayload(BaseModel):
    model_name: str
    dataset_name: str


class PredictPayload(BaseModel):
    data: List[
        List[float]
    ]  # Assuming data is provided as list of lists (rows of features)


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}


# Endpoint to select the model and dataset
@app.post("/set_model")
async def set_model(payload: SetModelPayload):
    global current_model, current_dataset, current_dataset_columns

    # Check if the requested model exists
    if payload.model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if the requested dataset exists
    if payload.dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load the model and dataset
    current_model = models[payload.model_name]
    current_dataset = pd.read_csv(datasets[payload.dataset_name])

    # Store the dataset columns for later use
    current_dataset_columns = current_dataset.columns.tolist()

    return {
        "message": f"Switched to {payload.model_name} model with {payload.dataset_name} dataset.",
        "dataset_columns": current_dataset_columns,
    }


# Endpoint to make predictions
@app.post("/predict")
async def predict(payload: PredictPayload):
    global current_model, current_dataset_columns

    if current_model is None:
        raise HTTPException(
            status_code=400, detail="Model is not set. Please set a model first."
        )

    if current_dataset_columns is None:
        raise HTTPException(
            status_code=400, detail="Dataset is not set. Please set a dataset first."
        )

    # Convert input data to DataFrame, ensuring column order matches the dataset
    input_data = pd.DataFrame(
        payload.data, columns=current_dataset_columns[:-1]
    )  # Exclude target column

    # Make predictions
    try:
        predictions = current_model.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return {"predictions": predictions.tolist()}


# Optional endpoint for explanation (if you want to add explainability)
@app.post("/explain")
async def explain(data: List[List[float]]):
    # You can use SHAP or LIME to generate explanations
    return {"explanation": "Not implemented yet"}


# To run the FastAPI app using uvicorn:
# uvicorn main:app --reload
