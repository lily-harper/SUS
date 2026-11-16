"""Evaluate the model selected in 2023 once on 2024 runs."""

import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED
from src.prep import make_preprocessor, make_xy


def main():
    """Input: saved 2023 choice and year split CSVs. Output: 2024 test metrics."""
    with open("outputs/selected_model_2023.json") as file:
        choice = json.load(file)
    if "params" not in choice:
        raise ValueError("Saved model choice has no parameters; rerun 03_run_models.py before testing.")

    training_rows = pd.concat(
        [pd.read_csv(f"data/{name}.csv") for name in
        ("train", "tune", "threshold", "selection")],
        ignore_index=True,
    )
    X_train, y_train = make_xy(training_rows)

    X_test, y_test = make_xy(pd.read_csv("data/test.csv"))

    models = {
        "logreg": ("logistic", LogisticRegression(max_iter=1000)),
        "tree": ("tree", DecisionTreeClassifier(random_state=SEED)),
        "forest": ("forest", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    }
    step_name, estimator = models[choice["model"]]
    model = Pipeline([("preprocess", make_preprocessor()), (step_name, estimator)])
    model.set_params(**choice["params"])
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities >= choice["threshold"]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    cost = fp + 2 * fn
    review_none_cost = 2 * int(y_test.sum())

    print(f"2024 test: {choice['model']} at threshold {choice['threshold']}")
    print("                       Normal queue  Priority review")
    print(f"Actual verified (TN/FP) {tn:12} {fp:16}")
    print(f"Actual rejected (FN/TP) {fn:12} {tp:16}")
    print(f"Model cost: {cost} ({1000 * cost / len(y_test):.2f} per 1,000 runs)")
    print(f"Review-none cost: {review_none_cost} ({1000 * review_none_cost / len(y_test):.2f} per 1,000 runs)")
    print(f"Model minus review none: {1000 * (cost - review_none_cost) / len(y_test):.2f} per 1,000 runs")


if __name__ == "__main__":
    main()
