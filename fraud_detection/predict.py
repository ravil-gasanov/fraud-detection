import os

from fastapi import FastAPI
import joblib
import pandas as pd

from fraud_detection.common import get_X

app = FastAPI()


def load_model():
    model = joblib.load("models/rfc_trained.pkl")

    return model


@app.get("/")
def root():
    return {"message": "To flag, or not to flag! That is the question."}


@app.post("/predict")
def predict(data: dict):
    try:
        data = pd.DataFrame(data)
        X = get_X(data=data)
    except Exception as e:
        return {"error": f"Invalid input data. Please check your input.\n{e}"}

    try:
        model = load_model()
    except FileNotFoundError:
        return {
            f'error": "Model not found. Please train the model first.\n{os.listdir("models/")}'
        }

    try:
        prediction = model.predict(X)[0]

        if prediction == 1:
            prediction = "Fraud"
        else:
            prediction = "Not Fraud"
    except Exception as e:
        return {"error": f"Prediction failed. Please check your model.\n{e}"}

    return {"prediction": prediction}
