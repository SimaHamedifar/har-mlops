# HAR-MLOps

MLOps pipeline for Human Activity Recognition using PyTorch.

## Overview
This project contains an API for serving predictions using a trained MuRAL BiLSTM + self-attention HAR checkpoint. It includes preprocessing, training scripts with MLflow tracking, and an inference API built with FastAPI.

## Installation
You can install the dependencies using `pip`:

```bash
pip install -e .[dev]
```

## Running the API
The API can be served locally using Uvicorn:

```bash
uvicorn activity_ml.api:app --host 0.0.0.0 --port 8000
```

Alternatively, you can build and run it using Docker:

```bash
docker build -t har-mlops .
docker run -p 8000:8000 har-mlops
```

## Testing
To run the automated tests, simply execute:

```bash
pytest tests/
```
