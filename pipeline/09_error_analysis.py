"""Compare 2024 review errors by game and first listed developer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.forest import make_forest
from src.models.logistic import make_log_reg
from src.models.tree import make_tree
from src.prep import make_xy


def summarize_slice(group, rng):
    """Input: one game's or developer's 2024 rows and a random generator. Output: counts, costs, and 95% intervals."""
    actual = group["rejected"].to_numpy() == 1
    review = group["priority_review"].to_numpy()
    fp = int((~actual & review).sum())
    fn = int((actual & ~review).sum())
    model_cost = (~actual & review).astype(int) + 2 * (actual & ~review).astype(int)
    none_cost = 2 * actual.astype(int)

    # Resample whole runs so both rules are compared on the same runs.
    model_samples = []
    difference_samples = []
    for _ in range(1000):
        sample = rng.integers(0, len(group), len(group))
        model_samples.append(1000 * model_cost[sample].mean())
        difference_samples.append(1000 * (model_cost[sample] - none_cost[sample]).mean())

    return {
        "runs": len(group),
        "rejected": int(actual.sum()),
        "review_rate": review.mean(),
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": int((actual & review).sum()),
        "true_negative": int((~actual & ~review).sum()),
        "model_cost_per_1000": 1000 * model_cost.mean(),
        "model_cost_ci_low": np.percentile(model_samples, 2.5),
        "model_cost_ci_high": np.percentile(model_samples, 97.5),
        "review_none_cost_per_1000": 1000 * none_cost.mean(),
        "difference_from_none": 1000 * (model_cost - none_cost).mean(),
        "difference_ci_low": np.percentile(difference_samples, 2.5),
        "difference_ci_high": np.percentile(difference_samples, 97.5),
    }


def main():
    """Input: saved model choice, parameters, and year CSVs. Output: 2024 game and developer slice CSVs."""
    choice = json.loads(Path("outputs/selected_model_2023.json").read_text())
    train = pd.read_csv("data/train.csv")
    tune = pd.read_csv("data/tune.csv")
    test = pd.read_csv("data/test.csv")

    X_train, y_train = make_xy(train)
    X_tune, y_tune = make_xy(tune)
    X_test, _ = make_xy(test)
    factories = {"logreg": make_log_reg, "tree": make_tree, "forest": make_forest}
    model = factories[choice["model"]](X_train, y_train, X_tune, y_tune)
    model.fit(X_train, y_train)
    test["priority_review"] = model.predict_proba(X_test)[:, 1] >= choice["threshold"]

    # A game with several listed developers belongs to its first listed developer.
    test["first_developer"] = (
        test["developerNames"].fillna("Unknown").str.split(",").str[0].str.strip().replace("", "Unknown")
    )

    rng = np.random.default_rng(67)
    for column, label in (("gameName", "game"), ("first_developer", "developer")):
        rows = []
        for name, group in test.groupby(column, dropna=False):
            rows.append({label: name, **summarize_slice(group, rng)})

        results = pd.DataFrame(rows).sort_values("difference_from_none", ascending=False)
        path = Path(f"outputs/error_slices_2024_{label}.csv")
        results.to_csv(path, index=False)
        print(f"Saved {path}")

        # Small slices can look extreme by chance; show the worst substantial slice.
        substantial = results.loc[results["runs"] >= 100]
        if not substantial.empty:
            worst = substantial.iloc[0]
            print(f"Worst {label} with at least 100 runs: {worst[label]}")
            print(f"  Runs: {worst['runs']}; rejected: {worst['rejected']}")
            print(f"  Model cost per 1,000: {worst['model_cost_per_1000']:.1f}")
            print(f"  Difference from review none: {worst['difference_from_none']:.1f} "
                  f"(95% interval {worst['difference_ci_low']:.1f} to {worst['difference_ci_high']:.1f})")


if __name__ == "__main__":
    main()
