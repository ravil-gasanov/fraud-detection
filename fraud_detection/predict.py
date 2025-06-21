from fastapi import FastAPI
from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd

from fraud_detection.common import get_X

app = FastAPI()

mlflow.set_tracking_uri("http://mlflow:5000")

logger.add(
    "logs/predict.log",
    rotation="1 MB",
    retention="10 days",
    level="INFO",
    format="{time} {level} {message}",
)


def load_model():
    model_name = "trained_rfc"
    try:
        logger.info("Connecting to MLflow tracking server at http://mlflow:5000")

        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/latest",
        )
        logger.info("Model loaded from MLflow: {}", model_name)
    except Exception:
        logger.error("Failed to load model. Ensure that the model is registered in MLflow.")
        return None

    return model


@app.get("/")
def root():
    return {"message": "To flag, or not to flag! That is the question."}


@app.post("/predict")
def predict(data: dict):
    logger.info("Received data for prediction: {}", data)

    data = pd.DataFrame(data)
    logger.info("Data converted to DataFrame:\n{}", data)

    X = get_X(data=data)
    logger.info("Did something to X:\n{}", X)

    model = load_model()
    logger.info("Model loaded successfully.")

    prediction = model.predict(X)[0]
    logger.info("Prediction made: {}", prediction)

    if prediction == 1:
        prediction = "Fraud"
    else:
        prediction = "Not Fraud"

    logger.info("Final prediction: {}", prediction)

    return {"prediction": prediction}
