import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_detection.features import FeatureTransformer


def get_X(data: pd.DataFrame) -> pd.DataFrame:
    X_columns = ["V4", "V11", "V7", "Amount"]

    return data[X_columns]


def get_y(data: pd.DataFrame) -> pd.Series:
    y_column = "Class"

    return data[y_column]


def build_pipeline(model_name, model):
    steps = [
        ("FeatureTransformer", FeatureTransformer()),
        (model_name, model),
    ]

    return Pipeline(steps=steps)
