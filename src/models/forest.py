import json
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import SEED, TRAIN_DATA, data_stuff
from src.models.thresholds import threshold_metrics
from src.models.tuning import PARAMS_PATH_FOREST, tune_forest
from src.prep import make_preprocessor
from src.prep import make_xy

def make_forest(X_train, y_train, X_val, y_val):
    """Input: train and validation features and labels. Output: an unfitted, tuned forest pipeline."""
    forest_pipeline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("forest", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ])

    if PARAMS_PATH_FOREST.exists():
        best_params = json.loads(PARAMS_PATH_FOREST.read_text(encoding="utf-8"))
    else:
        print("Tuning parameters")
        best_params = tune_forest(forest_pipeline, X_train, y_train, X_val, y_val)

    return forest_pipeline.set_params(**best_params)


def main():
    """Input: training CSV. Output: forest threshold results for 2022 validation."""
    X_train, y_train = make_xy(pd.read_csv("data/train.csv"))
    X_val, y_val = make_xy(pd.read_csv("data/tune.csv"))
    X_threshold, y_threshold = make_xy(pd.read_csv("data/threshold.csv"))

    model = make_forest(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_threshold)[:, 1]
    threshold_metrics(y_threshold, probabilities, save_path="forest")


if __name__ == "__main__":
    main()
