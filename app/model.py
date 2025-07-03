# Building and Deploying a Real-Time Prediction Pipeline
# This module ensures that input data comes in the expected format for the model

# Import BaseModel from Pydantic for data validation
from pydantic import BaseModel

# Define the input data class for the model
class InputData(BaseModel):
    """
    InputData defines the expected structure for incoming data to the prediction API.
    Each attribute corresponds to a feature required by the trained model.
    Using Pydantic's BaseModel ensures type validation and automatic error handling
    if the input data does not match the expected format.
    """
    # First input feature as a floating point number
    feature1: float

    # Second input feature as a floating point number
    feature2: float

    # Third input feature as a floating point number
    feature3: float
