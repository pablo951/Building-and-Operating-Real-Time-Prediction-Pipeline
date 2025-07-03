# Client module to consume the API

import requests

def api_test():
    """
    Function to test the health endpoint of the API.
    It sends a GET request to the /health endpoint and prints the response.
    This is useful to check if the API server is running and reachable.
    """
    # Endpoint URL
    url = "http://localhost:8000/health"

    # Send GET request to the API and store the response
    response = requests.get(url)

    # Print the response
    if response.status_code == 200:
        print("API Status Health: ", response.json())
    else:
        print(f"An error occurred during API health check. Error Status Code: {response.status_code}")

def predict_via_api():
    """
    Function to consume the prediction endpoint of the API.
    It sends a POST request with input features to the /predict endpoint and prints the prediction result.
    This function demonstrates how to interact with a machine learning model served via an API.
    """
    # Endpoint URL for prediction
    url = "http://localhost:8000/predict"

    # Input data (example features)
    payload = {
        "feature1": 1.4,
        "feature2": 2.7,
        "feature3": 3.5
    }

    # Send POST request to the API with the input data
    response = requests.post(url, json=payload)

    # Print the response
    if response.status_code == 200:
        print("API Status: ", response.json())
    else:
        print(f"API call failed. Status code: {response.status_code}")

# Main block
if __name__ == "__main__":
    # First, check if the API is healthy
    api_test()
    # Then, make a prediction request
