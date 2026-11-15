from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.prep import make_preprocessor


def make_model_pipeline(model):
    """
    Input:
        model: an unfitted sklearn estimator.

    Output:
        A preprocessing and model pipeline.
    """
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", model),
    ])


def make_decision_tree_pipeline():
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=50,
        random_state=0,
    )
    return make_model_pipeline(model)


def make_random_forest_pipeline():
    model = RandomForestClassifier(
        n_estimators=20,
        random_state=0,
        n_jobs=-1,
    )
    return make_model_pipeline(model)
