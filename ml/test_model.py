"""Unit tests for the data and model functions."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.config import CAT_FEATURES, LABEL
from ml.data import load_data, process_data, split_data
from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture(scope="module")
def processed():
    train_set, test_set = split_data(load_data())
    X_train, y_train, encoder, lb = process_data(
        train_set, CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test_set, CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb
    )
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def model(processed):
    X_train, y_train, _, _ = processed
    return train_model(X_train, y_train)


def test_process_data_shapes(processed):
    X_train, y_train, X_test, y_test = processed
    assert X_train.shape[0] == len(y_train)
    assert X_train.shape[1] == X_test.shape[1]
    assert set(np.unique(y_train)) <= {0, 1}


def test_inference_mode_without_label():
    df = load_data().head(5).drop(columns=[LABEL])
    _, _, encoder, _ = process_data(load_data(), CAT_FEATURES, label=LABEL, training=True)
    X, y, _, _ = process_data(df, CAT_FEATURES, training=False, encoder=encoder)
    assert X.shape[0] == 5 and y.size == 0


def test_train_model(model):
    assert isinstance(model, RandomForestClassifier)


def test_inference(model, processed):
    _, _, X_test, _ = processed
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)
    assert np.isin(preds, [0, 1]).all()


def test_compute_model_metrics(model, processed):
    _, _, X_test, y_test = processed
    metrics = compute_model_metrics(y_test, inference(model, X_test))
    assert len(metrics) == 3
    assert all(0 <= m <= 1 for m in metrics)
