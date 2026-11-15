"""Evaluate the model selected in 2023 once on 2024 runs."""

import json
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.prep import make_xy
from src.models.logistic import make_log_reg
from src.models.tree import make_tree
from src.models.forest import make_forest


def main():
    """Input: saved 2023 choice and year split CSVs. Output: 2024 test metrics."""
    with open("outputs/selected_model_2023.json") as file:
        choice = json.load(file)

    X_train, y_train = make_xy(pd.read_csv("data/train.csv"))
    X_tune, y_tune = make_xy(pd.read_csv("data/tune.csv"))
    X_test, y_test = make_xy(pd.read_csv("data/test.csv"))

    models = {"logreg": make_log_reg, "tree": make_tree, "forest": make_forest}
    model = models[choice["model"]](X_train, y_train, X_tune, y_tune)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities >= choice["threshold"]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    cost = fp + 2 * fn

    print(f"2024 test: {choice['model']} at threshold {choice['threshold']}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Cost: {cost} ({1000 * cost / len(y_test):.2f} per 1,000 runs)")


if __name__ == "__main__":
    main()
