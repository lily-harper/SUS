from sklearn.model_selection import GridSearchCV
import pandas as pd 
import json

from pathlib import Path

from src.config import PARAMS_PATH, tune_set

PARAMS_PATH_LOG = PARAMS_PATH / "logreg_2021_best_params.json"
PARAMS_PATH_TREE = PARAMS_PATH / "tree_2021_best_params.json"
PARAMS_PATH_FOREST = PARAMS_PATH / "forest_2021_best_params.json"


def tune_logreg(log_reg, X_train, y_train, X_val, y_val):
    X_tuning, y_tuning, validation_fold, splitter = tune_set(X_train, y_train, X_val, y_val)

    parameter_grid = {
        "logistic__C": [0.01, 0.1, 1, 10],
        "logistic__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator = log_reg,
        param_grid = parameter_grid,
        cv = splitter,
        scoring="neg_log_loss",
        refit = False,
    )
    grid_search.fit(X_tuning, y_tuning)

    PARAMS_PATH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PARAMS_PATH_LOG.open("w") as f:
        json.dump(grid_search.best_params_, f, indent = 2)

    return grid_search.best_params_

def tune_tree(tree_pipeline, X_train, y_train, X_val, y_val):
    X_tuning, y_tuning, validation_fold, splitter = tune_set(X_train, y_train, X_val, y_val)

    tree_search = GridSearchCV(
        estimator=tree_pipeline,
        param_grid={
            "tree__max_depth": [3, 4],
            "tree__min_samples_leaf": [5, 20, 35, 50],
            "tree__class_weight": [None, "balanced"],
        },
        cv=splitter,
        scoring="neg_log_loss",
        refit=False,
    )

    tree_search.fit(X_tuning, y_tuning)

    PARAMS_PATH_TREE.parent.mkdir(parents=True, exist_ok=True)
    with PARAMS_PATH_TREE.open("w") as f:
        json.dump(tree_search.best_params_, f, indent = 2)

    return tree_search.best_params_


def tune_forest(forest_pipeline, X_train, y_train, X_val, y_val):
    """Input: forest pipeline and train/validation data. Output: best forest parameters."""
    X_tuning, y_tuning, validation_fold, splitter = tune_set(X_train, y_train, X_val, y_val)

    forest_search = GridSearchCV(
        estimator=forest_pipeline,
        param_grid={
            "forest__max_depth": [10, None],
            "forest__min_samples_leaf": [5, 20],
            "forest__class_weight": [None, "balanced"],
        },
        cv=splitter,
        scoring="neg_log_loss",
        refit=False,
    )
    forest_search.fit(X_tuning, y_tuning)

    PARAMS_PATH_FOREST.parent.mkdir(parents=True, exist_ok=True)
    with PARAMS_PATH_FOREST.open("w") as f:
        json.dump(forest_search.best_params_, f, indent=2)

    return forest_search.best_params_

