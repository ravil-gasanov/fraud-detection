from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd

from fraud_detection.common import get_X

app = FastAPI()

mlflow.set_tracking_uri("http://localhost:5000")


def load_model():
    model_name = "trained_rfc"
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/latest",
    )

    return model


@app.get("/")
def root():
    return {"message": "To flag, or not to flag! That is the question."}


@app.post("/predict")
def predict(data: dict):
    data = pd.DataFrame(data)
    X = get_X(data=data)
    model = load_model()

    prediction = model.predict(X)[0]

    if prediction == 1:
        prediction = "Fraud"
    else:
        prediction = "Not Fraud"

    return {"prediction": prediction}
