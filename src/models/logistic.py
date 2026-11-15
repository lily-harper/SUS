import pandas as pd 
import numpy as np
from src.prep import make_xy
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.prep import make_preprocessor

import json
from pathlib import Path

from src.models.tuning import PARAMS_PATH_LOG

from src.models.tuning import tune_logreg
from src.models.thresholds import threshold_metrics

def make_log_reg(X_train, y_train, X_val, y_val):

    log_reg = Pipeline([
        ("preprocess", make_preprocessor()),
        ("logistic", LogisticRegression(max_iter = 1000)),
    ])

    if PARAMS_PATH_LOG.exists():
        best_params = json.loads(PARAMS_PATH_LOG.read_text(encoding="utf-8"))
    else:
        print("Tuning parameters")
        best_params = tune_logreg(log_reg, X_train, y_train, X_val, y_val)
        PARAMS_PATH_LOG.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    return log_reg.set_params(**best_params)

def main():
    X_train, y_train = make_xy(pd.read_csv("data/train.csv"))
    X_val, y_val = make_xy(pd.read_csv("data/tune.csv")) 
    X_threshold, y_threshold = make_xy(pd.read_csv("data/threshold.csv"))

    model = make_log_reg(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_threshold)[:, 1]
    threshold_metrics(y_threshold, probabilities, save_path="logreg")

if __name__ == "__main__":
    main()
