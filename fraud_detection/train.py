import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from fraud_detection.config import TRAIN_PATH, RANDOM_STATE
from fraud_detection.common import build_pipeline, get_X, get_y


def train():
    data = pd.read_csv(TRAIN_PATH)

    X = get_X(data=data)
    y = get_y(data=data)

    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    pipeline = build_pipeline(model_name="rfc", model=rfc)

    print("Training...")
    pipeline.fit(X=X, y=y)

    y_pred = pipeline.predict(X=X)
    score = f1_score(y_true=y, y_pred=y_pred)
    print(f"Training f1-score: {score}")

    with open("./models/rfc_trained.pkl", "wb") as f:
        pickle.dump(pipeline, f)


if __name__ == "__main__":
    train()
