#  Building and Deploying a Real-Time Prediction Pipeline

# Import pandas library for data manipulation
import pandas as pd

# Import LinearRegression class from scikit-learn
from sklearn.linear_model import LinearRegression

# Import function to split data into train and test sets
from sklearn.model_selection import train_test_split

# Import custom function to save the model and metadata
from app.utils import save_model_and_metadata

def model_training(data_path="dados/dataset.csv"):
    """
    Main function to train a regression model.
    Loads data, splits into features and target, trains a linear regression model,
    and saves the trained model along with its metadata.

    Args:
        data_path (str): Path to the CSV file containing the dataset.
    """
    # Load data from a CSV file
    df = pd.read_csv(data_path)

    # Separate features (X) from target (y)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the linear regression model
    model = LinearRegression()

    # Train the model with the training data
    model.fit(X_train, y_train)

    # Save the trained model and its metadata, returning the model version
    version = save_model_and_metadata(model, X_train, y_train)

    # Print the version of the trained model
    print(f"\nTrained model saved with version {version}\n")

# Run the training function if the script is executed directly
if __name__ == "__main__":
    model_training()