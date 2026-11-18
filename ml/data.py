"""Data loading and preprocessing."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from ml.config import DATA_PATH, LABEL


def load_data(path=DATA_PATH):
    """Load the census CSV, strip whitespace from headers and values, drop duplicates."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df.drop_duplicates()


def split_data(df, test_size=0.20, random_state=42):
    """Stratified train/test split on the label."""
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[LABEL]
    )


def process_data(
    X, categorical_features=None, label=None, training=True, encoder=None, lb=None
):
    """Process data for training or inference.

    One-hot encodes the categorical features and binarizes the label.

    Parameters
    ----------
    X : pd.DataFrame
        Features, plus the label column if `label` is given.
    categorical_features : list[str]
        Names of the categorical columns.
    label : str or None
        Name of the label column. If None, `y` is returned empty (inference).
    training : bool
        If True, fit a new encoder and label binarizer; otherwise use the ones passed in.
    encoder : OneHotEncoder
        Fitted encoder, required when training is False.
    lb : LabelBinarizer
        Fitted label binarizer, used when training is False and a label is given.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    encoder : OneHotEncoder
    lb : LabelBinarizer
    """
    categorical_features = categorical_features or []

    if label is not None:
        y = X[label]
        X = X.drop(columns=[label])
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features).values

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        if encoder is None:
            raise ValueError("A fitted encoder is required when training=False.")
        X_categorical = encoder.transform(X_categorical)
        if label is not None and lb is not None:
            y = lb.transform(y.values).ravel()

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
