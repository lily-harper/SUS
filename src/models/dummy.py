from sklearn.dummy import DummyClassifier

from src.config import TRAIN_DATA, data_stuff
from src.models.thresholds import threshold_metrics


def make_dummy():
    """Input: none. Output: an unfitted classifier using the training class rate."""
    return DummyClassifier(strategy="most_frequent")


def main():
    """Input: training data at TRAIN_DATA. Output: saved validation threshold metrics."""
    X_train, y_train, X_val, y_val = data_stuff(TRAIN_DATA)

    model = make_dummy()
    model.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_val)

if __name__ == "__main__":
    main()
