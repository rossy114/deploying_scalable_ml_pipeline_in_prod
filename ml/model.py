"""Model training, evaluation and persistence."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.config import MODEL_DIR


def train_model(X_train, y_train, random_state=42):
    """Train a random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=20,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """Return (precision, recall, F1) for binary labels."""
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inference and return predictions."""
    return model.predict(X)


def save_artifacts(model, encoder, lb, model_dir=MODEL_DIR):
    """Save the model, encoder and label binarizer."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib", compress=3)
    joblib.dump(encoder, model_dir / "encoder.joblib")
    joblib.dump(lb, model_dir / "lb.joblib")


def load_artifacts(model_dir=MODEL_DIR):
    """Load the model, encoder and label binarizer."""
    return (
        joblib.load(model_dir / "model.joblib"),
        joblib.load(model_dir / "encoder.joblib"),
        joblib.load(model_dir / "lb.joblib"),
    )
