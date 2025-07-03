# Project: Building and Operating a Real-Time Prediction Pipeline

This project implements a machine learning model versioning pipeline.  
It is designed for lightweight models, where the deployment process allows version switching **without stopping the system** — depending on the model's size.

---

## Getting Started

### 1. Ensure Docker Desktop is installed and running on your system.

---

### 2. Open a terminal, navigate to the project folder, and build the Docker image:

```bash
docker compose build --no-cache

---
### 3. In the same terminal, run the following command to train the model and save its metadata:.

docker compose run --rm app python -m app.train


### 4 Open a new terminal, navigate to the same project folder, and start the API container:

docker compose up --build


### 5. Open another (yes, a third) terminal and execute the following to consume the API:

python client.py

