import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_metrics(model, test, feature_columns, threshold=1 / 3,
                     review_cost=1, missed_rejection_cost=2):
    """Score a fitted model; positive class means priority review/rejected run."""
    if test.empty:
        raise ValueError("Cannot evaluate an empty test set")

    probabilities = model.predict_proba(test[feature_columns])[:, 1]
    actual = test["rejected"].to_numpy()
    priority_review = probabilities > threshold
    tn, fp, fn, tp = confusion_matrix(actual, priority_review, labels=[0, 1]).ravel()

    realized_cost = review_cost * fp + missed_rejection_cost * fn
    expected_cost = np.where(
        priority_review,
        review_cost * (1 - probabilities),
        missed_rejection_cost * probabilities,
    ).sum()

    return {
        "n_runs": len(test),
        "accuracy": accuracy_score(actual, priority_review),
        "precision": precision_score(actual, priority_review, zero_division=0),
        "recall": recall_score(actual, priority_review, zero_division=0),
        "f1": f1_score(actual, priority_review, zero_division=0),
        "review_rate": priority_review.mean(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "realized_cost_per_1000": 1000 * realized_cost / len(test),
        "expected_cost_per_1000": 1000 * expected_cost / len(test),
    }

def evaluate(model, test, feature_columns):
    return evaluate_metrics(model, test, feature_columns)["realized_cost_per_1000"]


def evaluate_baselines(test, review_cost=1, missed_rejection_cost=2):
    """Return metrics for the review-none and review-all decision rules."""
    if test.empty:
        raise ValueError("Cannot evaluate an empty test set")

    rejected = test["rejected"].to_numpy()
    n_rejected = int(rejected.sum())
    n_verified = len(test) - n_rejected

    return pd.DataFrame([
        {
            "baseline": "review_none",
            "n_runs": len(test),
            "accuracy": n_verified / len(test),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "review_rate": 0.0,
            "tn": n_verified, "fp": 0, "fn": n_rejected, "tp": 0,
            "realized_cost_per_1000": (
                1000 * missed_rejection_cost * n_rejected / len(test)
            ),
        },
        {
            "baseline": "review_all",
            "n_runs": len(test),
            "accuracy": n_rejected / len(test),
            "precision": n_rejected / len(test),
            "recall": 1.0 if n_rejected else 0.0,
            "f1": 2 * n_rejected / (len(test) + n_rejected) if n_rejected else 0.0,
            "review_rate": 1.0,
            "tn": 0, "fp": n_verified, "fn": 0, "tp": n_rejected,
            "realized_cost_per_1000": 1000 * review_cost * n_verified / len(test),
        },
    ])


def evaluate_years(df, feature_columns, make_model, years = (2021, 2022, 2023)):
    results = []

    for year in years:
        train = df[df["date"].dt.year < year]
        validation = df[df["date"].dt.year == year]

        model = make_model()
        model.fit(train[feature_columns], train["rejected"])

        results.append({
            "year": year,
            "cost_per_1000": evaluate(model, validation, feature_columns)
        })

    return pd.DataFrame(results)
