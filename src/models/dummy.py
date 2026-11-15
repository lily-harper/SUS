from sklearn.dummy import DummyClassifier

from src.config import TRAIN_DATA, data_stuff
from src.models.thresholds import threshold_metrics


def make_dummy():
    """Input: none. Output: an unfitted classifier using the training class rate."""
    return DummyClassifier(strategy="prior")


def main():
    """Input: training data at TRAIN_DATA. Output: saved validation threshold metrics."""
    X_train, y_train, X_val, y_val = data_stuff(TRAIN_DATA)

    model = make_dummy()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    threshold_metrics(y_val, probabilities, save_path="dummy")


if __name__ == "__main__":
    main()
