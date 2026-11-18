"""Compute model performance on slices of each categorical feature.

Writes one line per (feature, value) slice of the test set to slice_output.txt.
Run from the repository root after training:  python evaluate.py
"""
from ml.config import CAT_FEATURES, LABEL, ROOT
from ml.data import load_data, process_data, split_data
from ml.model import compute_model_metrics, inference, load_artifacts


def slice_metrics(df, model, encoder, lb):
    """Yield (feature, value, n, precision, recall, F1) for every slice of df."""
    for feature in CAT_FEATURES:
        for value in sorted(df[feature].unique()):
            subset = df[df[feature] == value]
            X, y, _, _ = process_data(
                subset, CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb
            )
            yield (feature, value, len(subset), *compute_model_metrics(y, inference(model, X)))


def main():
    model, encoder, lb = load_artifacts()
    _, test_set = split_data(load_data())

    X, y, _, _ = process_data(
        test_set, CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb
    )
    precision, recall, fbeta = compute_model_metrics(y, inference(model, X))
    print(f"Overall  precision: {precision:.3f}  recall: {recall:.3f}  F1: {fbeta:.3f}")

    with open(ROOT / "slice_output.txt", "w") as out:
        for feature, value, n, p, r, f in slice_metrics(test_set, model, encoder, lb):
            out.write(f"[{feature} = {value}] n={n}  precision={p:.3f}  recall={r:.3f}  F1={f:.3f}\n")
    print("Slice metrics written to slice_output.txt")


if __name__ == "__main__":
    main()
