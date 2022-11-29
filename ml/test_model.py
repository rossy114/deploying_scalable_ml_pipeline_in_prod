import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
# from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier
import logging
import os


path="/"

df = pd.read_csv(os.path.join(path, "data/raw/census.csv"))
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

train_set, test_set = train_test_split(df, test_size=0.20, random_state=42, stratify=df.salary)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


X_train, y_train, encoder, lb = process_data(
train_set, categorical_features=cat_features, label="salary", training=True
    )


def test_train_model():
    #Testfunction will returns a trained RandomForestClassifier
    model = train_model(X_train, y_train)
    assert type(model) == RandomForestClassifier

def test_compute_model_metrics():
    #Test this will return the three metrics and each ranges from 0 to 1
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    assert len(metrics) == 3
    assert type(metrics) == tuple
    for metric in metrics:
        assert metric >=0 and metric <= 1


def test_inference():
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(X_train) #Assert that the length is the same as x_train
    assert np.all((preds==0)|(preds == 1)) == True #Assert all values are 0 or 1


if __name__ == "__main__":
    test_train_model()
    test_compute_model_metrics()
    test_inference()