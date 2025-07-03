# Building and Deploying a Real-Time Prediction Pipeline

import pandas as pd
from fastapi import FastAPI
# Import the input data definition class for the model
from app.model import InputData
# Import the custom function to load the model and its metadata
from app.utils import load_model_and_metadata

# Create an instance of the FastAPI application
app = FastAPI()

# Define a health check route
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the API service is running.
    Returns a simple status message.
    """
    return {"status": "ok"}

# Define a prediction route
@app.post("/predict")
def predict(input_data: InputData):
    """
    Prediction endpoint that receives input data, loads the latest trained model,
    and returns the prediction along with model metadata.

    Args:
        input_data (InputData): Input features for the model.

    Returns:
        dict: Prediction result and model metadata.
    """
    # Load the trained model and its metadata
    model, metadata = load_model_and_metadata()

    # Check if a model is available
    if model is None:
        # Return an error if no trained model is available
        return {"error": "No trained model available"}

    # Create a DataFrame from the input data
    X = pd.DataFrame([[input_data.feature1, input_data.feature2, input_data.feature3]],
                     columns=["feature1", "feature2", "feature3"])

    # Make predictions using the loaded model
    prediction = model.predict(X)[0]

    # Return the prediction and the model version used
    return {
        "prediction": prediction,
        "model_version": metadata["version"],
        "scikit_learn_version": metadata["scikit_learn_version"],
        "r2_train": metadata["r2_train"]
    }