import shutil

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import RANDOM_STATE, RAW_PATH, TEST_PATH, TEST_SIZE, TRAIN_PATH


def download_dataset():
    """
    Downloads the credit card fraud detection dataset from Kaggle.
    """
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print(f"Path: {path}")
    shutil.move(path + "/creditcard.csv", RAW_PATH)

    print(f"Dataset downloaded and moved to {RAW_PATH}")


def split_dataset():
    """
    Splits the dataset into train and test sets.
    """
    data = pd.read_csv(RAW_PATH)
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print("Dataset split into train and test sets.")


if __name__ == "__main__":
    download_dataset()
    split_dataset()
