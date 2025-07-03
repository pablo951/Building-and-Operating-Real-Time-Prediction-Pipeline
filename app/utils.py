# - Building and Deploying a Real-Time Prediction Pipeline

import json  # Library for handling JSON files
import os    # Library for interacting with the operating system
import pickle  # Library for serializing and deserializing Python objects
from datetime import datetime  # For working with timestamps
import sklearn  # To get the scikit-learn version

def save_model_and_metadata(model, X_train, y_train, model_dir="modelo"):
    """
    Saves a trained model and its metadata to disk.
    The function automatically manages model versioning by incrementing the version number.
    It also saves training metrics and environment information for reproducibility.

    Args:
        model: Trained scikit-learn model object.
        X_train: Training features used to fit the model.
        y_train: Training target used to fit the model.
        model_dir: Directory where the model and metadata will be saved.

    Returns:
        current_version (int): The version number assigned to the saved model.
    """
    # List all model files in the specified directory
    models = [f for f in os.listdir(model_dir) if f.startswith("model_v") and f.endswith(".pkl")]

    # Determine the next model version
    current_version = 1 if not models else max([int(m.split("_v")[-1].split(".pkl")[0]) for m in models]) + 1

    # Define file paths for the model and metadata
    model_file = os.path.join(model_dir, f"model_v{current_version}.pkl")
    metadata_file = os.path.join(model_dir, f"metadata_v{current_version}.json")

    # Save the model to the specified file
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Calculate R² score on the training set
    r2_train = model.score(X_train, y_train)

    # Generate metadata including version, timestamp, scikit-learn version, and training R²
    metadata = {
        "version": current_version,
        "timestamp": datetime.now().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "r2_train": r2_train
    }

    # Save metadata to the specified file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    return current_version

def load_model_and_metadata(model_dir="modelo"):
    """
    Loads the most recent model and its metadata from disk.
    This function helps in retrieving the latest trained model for inference or further evaluation.

    Args:
        model_dir: Directory where models and metadata are stored.

    Returns:
        model: The most recent trained model object, or None if no model exists.
        metadata: The metadata dictionary associated with the model, or None if no model exists.
    """
    # List all model files in the specified directory
    models = [f for f in os.listdir(model_dir) if f.startswith("model_v") and f.endswith(".pkl")]

    # Return None if no models exist in the directory
    if not models:
        return None, None

    # Extract available model versions
    versions = [int(m.split("_v")[-1].split(".pkl")[0]) for m in models]

    # Determine the latest version
    latest = max(versions)

    # Define file paths for the latest model and metadata
    model_file = os.path.join(model_dir, f"model_v{latest}.pkl")
    metadata_file = os.path.join(model_dir, f"metadata_v{latest}.json")

    # Load the model from file
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load the metadata from file
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return model, metadata