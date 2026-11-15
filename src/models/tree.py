import pandas as pd 
import numpy as np

from src.config import TRAIN_DATA, SEED, data_stuff
from src.prep import make_xy
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.prep import make_preprocessor

import json

from src.models.tuning import PARAMS_PATH_TREE

from src.models.tuning import tune_tree
from src.models.thresholds import threshold_metrics

def make_tree(X_train, y_train, X_val, y_val):

    tree_pipeline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("tree", DecisionTreeClassifier(random_state=SEED)),
    ])

    if PARAMS_PATH_TREE.exists():
        best_params = json.loads(PARAMS_PATH_TREE.read_text(encoding="utf-8"))
    else:
        print("Tuning parameters")
        best_params = tune_tree(tree_pipeline, X_train, y_train, X_val, y_val)
        PARAMS_PATH_TREE.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    return tree_pipeline.set_params(**best_params)

def main():
    X_train, y_train = make_xy(pd.read_csv("data/train.csv"))
    X_val, y_val = make_xy(pd.read_csv("data/tune.csv"))    
    X_threshold, y_threshold = make_xy(pd.read_csv("data/threshold.csv"))

    model = make_tree(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_threshold)[:, 1]

    threshold_metrics(y_threshold, probabilities, save_path = "tree")

if __name__ == "__main__":
    main()
