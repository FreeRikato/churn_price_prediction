from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Global variables to store current model and dataset
current_model = None
current_dataset = None

# Load your models and datasets (pre-trained models)
models = {
    "logistic_regression": joblib.load("./models/logistic_regression_model.pkl"),
    "random_forest": joblib.load("./models/random_forest_model.pkl"),
}
datasets = {
    "bank": "datasets/BankChurners.csv",
    "ecommerce": "datasets/E Commerce Dataset.csv",
    "internet_service": "datasets/internet_service_churn.csv",
    "telecom": "datasets/orange_telecom.csv",
}


class setModelPayload(BaseModel):
    model_name: str
    dataset_name: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}


# Endpoint to select the model and dataset
@app.post("/set_model")
async def set_model(payload: setModelPayload):
    global current_model, current_dataset
    print(f"{payload.model_name=}/n{payload.dataset_name=}")
    if payload.model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    if payload.dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    current_model = models[payload.model_name]
    current_dataset = datasets[payload.dataset_name]

    return {"message": f"Switched to {payload.model_name} with {payload.dataset_name}"}


# Endpoint to make predictions
@app.post("/predict")
async def predict(data: list):
    global current_model

    if current_model is None:
        raise HTTPException(
            status_code=400, detail="Model is not set. Please set a model first."
        )

    # Convert input data to DataFrame (assuming it's passed as a list of lists)
    input_data = pd.DataFrame(data, columns=current_dataset.columns)

    # Make predictions
    predictions = current_model.predict(input_data)

    return {"predictions": predictions.tolist()}


# Optional endpoint for explanation (if you want to add explainability)
@app.post("/explain")
async def explain(data: list):
    # You can use SHAP or LIME to generate explanations
    return {"explanation": "Not implemented yet"}


# Run the FastAPI app using uvicorn:
# uvicorn main:app --reload
