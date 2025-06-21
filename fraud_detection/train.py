import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from fraud_detection.common import build_pipeline, get_X, get_y
from fraud_detection.config import RANDOM_STATE, TEST_PATH, TRAIN_PATH

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud_detection_experiment")


def train():
    data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    X = get_X(data=data)
    y = get_y(data=data)

    test_X = get_X(data=test_data)
    test_y = get_y(data=test_data)

    with mlflow.start_run():
        model_name = "trained_rfc"
        rfc = RandomForestClassifier(random_state=RANDOM_STATE)
        pipeline = build_pipeline(model_name=model_name, model=rfc)

        pipeline.fit(X=X, y=y)

        y_pred = pipeline.predict(X=test_X)
        test_f1 = f1_score(y_true=test_y, y_pred=y_pred)

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.sklearn.log_model(pipeline, artifact_path=model_name)


if __name__ == "__main__":
    train()
