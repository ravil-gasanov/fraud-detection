import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from fraud_detection.common import build_pipeline, get_X, get_y
from fraud_detection.config import RANDOM_STATE, TRAIN_PATH


def get_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def get_models():
    models = [
        ("logreg", LogisticRegression()),
        ("rfc", RandomForestClassifier()),
    ]

    return models


def get_model_params(model_name):
    params = {
        "logreg": [
            {
                "logreg__penalty": ["l1", "l2", "elasticnet"],
                "logreg__max_iter": [100, 500, 1000],
                "logreg__solver": ["liblinear"],
            },
        ],
        "rfc": [{}],
    }

    return params[model_name]


def run_experiment(train_path=TRAIN_PATH):
    data = pd.read_csv(train_path)

    X = get_X(data=data)
    y = get_y(data=data)

    cv = get_cv()

    models = get_models()

    for model_name, model in models:
        pipeline = build_pipeline(model_name=model_name, model=model)
        param_grid = get_model_params(model_name=model_name)

        gridcv = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
        )

        gridcv.fit(X=X, y=y)

        print(f"Best estimator: {gridcv.best_estimator_}")
        print(f"Mean test f1-score: {gridcv.best_score_}")


if __name__ == "__main__":
    run_experiment()
