import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.pipeline import Pipeline

from src.prep import make_preprocessor
from src.config import data_stuff, TRAIN_DATA
from src.prep import make_xy


def make_isolation(X_train, y_train, X_val, y_val):
    isolation_forest = Pipeline([
        ("preprocess", make_preprocessor()),
        ("forest", IsolationForest(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    isolation_forest.fit(X_train.loc[y_train == 0])

# Flip the sign so higher scores mean more unusual.
    val_scores = -isolation_forest.score_samples(X_val)

    # Try cutoffs across the validation score distribution.
    thresholds = np.quantile(val_scores, np.linspace(0, 1, 101))
    best_threshold = max(
        thresholds,
        key=lambda cutoff: f1_score(
            y_val, val_scores >= cutoff, average="macro"
        ),
    )

    test_scores = -isolation_forest.score_samples(X_val)
    test_predictions = (test_scores >= best_threshold).astype(int)

    print(f"Validation threshold:   {best_threshold:.4f}")
    print(f"Threshold accuracy:          {accuracy_score(y_val, test_predictions):.4f}")
    print(f"Threshold macro F1:          {f1_score(y_val, test_predictions, average='macro'):.4f}")
    print(f"Threshold average precision: {average_precision_score(y_val, test_scores):.4f}")
    return isolation_forest 

def main():
    X_train, y_train = make_xy(pd.read_csv("data/train.csv"))
    X_val, y_val = make_xy(pd.read_csv("data/threshold.csv"))    

    isolation_forest = make_isolation(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()
