from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.prep import make_preprocessor

# =========================================
# supervised model pipeline
# =========================================

def make_model_pipeline(model):
    """
    Input:
        model: an unfitted sklearn estimator.

    Output:
        A pipeline that preprocesses columns and then fits the model.
    """
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", model),
    ])

def make_logistic_pipeline(C=1.0, class_weight=None):
    """
    Output:
        An unfitted preprocessing-plus-logistic-regression pipeline.
    """
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        random_state=0,
    )

    return make_model_pipeline(model)

def make_lda_pipeline():
    model = LinearDiscriminantAnalysis()
    return make_model_pipeline(model)

def make_random_forest_pipeline():
    model = RandomForestClassifier(
        n_estimators = 200,
        random_state = 0,
        n_jobs= -1
    )
    return make_model_pipeline(model)


