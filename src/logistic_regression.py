import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline

from src.preprocessing import make_preprocessor

from paths import PREPROCESSED_RUNS 

def make_logistic():
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("classifier", LogisticRegression(max_iter = 1000)),
    ])