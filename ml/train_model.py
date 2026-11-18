"""Train the model and save it with its encoder and label binarizer.

Run from the repository root:  python -m ml.train_model
"""
from ml.config import CAT_FEATURES, LABEL
from ml.data import load_data, process_data, split_data
from ml.model import compute_model_metrics, inference, save_artifacts, train_model


def main():
    train_set, test_set = split_data(load_data())

    X_train, y_train, encoder, lb = process_data(
        train_set, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    model = train_model(X_train, y_train)
    save_artifacts(model, encoder, lb)

    X_test, y_test, _, _ = process_data(
        test_set, CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb
    )
    precision, recall, fbeta = compute_model_metrics(y_test, inference(model, X_test))
    print(f"Test set  precision: {precision:.3f}  recall: {recall:.3f}  F1: {fbeta:.3f}")


if __name__ == "__main__":
    main()
