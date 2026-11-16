"""Compare 2023 review costs under several missed-rejection penalties."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.models.logistic import make_log_reg
from src.models.tree import make_tree
from src.models.forest import make_forest
from src.models.tuning import PARAMS_PATH_LOG, PARAMS_PATH_TREE, PARAMS_PATH_FOREST
from src.prep import make_xy


THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]
MULTIPLIERS = [2, 5, 10]
COLUMNS = ["multiplier", "model", "threshold", "f1_filtered", "fp", "fn", "tp", "tn", "cost_per_1000"]


def count_results(y_true, predictions):
    """Input: true labels and 0/1 predictions. Output: TN, FP, FN, and TP counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def main():
    """Input: saved model parameters and four year-split CSVs. Output: a printed table and one sensitivity CSV."""
    # The model factories tune when a parameter file is missing. Require the saved
    # 2021 choices so this analysis does not change any model parameters.
    for path in (PARAMS_PATH_LOG, PARAMS_PATH_TREE, PARAMS_PATH_FOREST):
        if not path.exists():
            raise FileNotFoundError(f"Run the 2021 tuning stage first: {path}")

    X_train, y_train = make_xy(pd.read_csv("data/train.csv"))
    X_tune, y_tune = make_xy(pd.read_csv("data/tune.csv"))
    X_threshold, y_threshold = make_xy(pd.read_csv("data/threshold.csv"))
    X_selection, y_selection = make_xy(pd.read_csv("data/selection.csv"))

    models = {
        "logreg": make_log_reg,
        "tree": make_tree,
        "forest": make_forest,
    }
    probabilities = {}
    for name, make_model in models.items():
        model = make_model(X_train, y_train, X_tune, y_tune)
        model.fit(X_train, y_train)
        probabilities[name] = (
            model.predict_proba(X_threshold)[:, 1],
            model.predict_proba(X_selection)[:, 1],
        )

    rows = []
    for multiplier in MULTIPLIERS:
        for name, (threshold_scores, selection_scores) in probabilities.items():
            choices = []
            for threshold in THRESHOLDS:
                predictions = threshold_scores >= threshold
                tn, fp, fn, tp = count_results(y_threshold, predictions)
                choices.append({
                    "threshold": threshold,
                    "cost": fp + multiplier * fn,
                    "f1": f1_score(y_threshold, predictions, zero_division=0),
                })

            choice_table = pd.DataFrame(choices)
            for f1_filtered in (False, True):
                eligible = choice_table.loc[choice_table["f1"] > 0.1] if f1_filtered else choice_table
                if eligible.empty:
                    rows.append({
                        "multiplier": multiplier, "model": name, "threshold": None,
                        "f1_filtered": f1_filtered, "fp": None, "fn": None,
                        "tp": None, "tn": None, "cost_per_1000": None,
                    })
                    continue

                best = eligible.sort_values(["cost", "threshold"]).iloc[0]
                predictions = selection_scores >= best["threshold"]
                tn, fp, fn, tp = count_results(y_selection, predictions)
                rows.append({
                    "multiplier": multiplier,
                    "model": name,
                    "threshold": best["threshold"],
                    "f1_filtered": f1_filtered,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "tn": tn,
                    "cost_per_1000": 1000 * (fp + multiplier * fn) / len(y_selection),
                })

        # The two fixed rules have no probability threshold or F1 filter.
        for name, predictions in (
            ("review_none", y_selection * 0),
            ("review_all", y_selection * 0 + 1),
        ):
            tn, fp, fn, tp = count_results(y_selection, predictions)
            rows.append({
                "multiplier": multiplier,
                "model": name,
                "threshold": None,
                "f1_filtered": None,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tn": tn,
                "cost_per_1000": 1000 * (fp + multiplier * fn) / len(y_selection),
            })

    results = pd.DataFrame(rows, columns=COLUMNS)
    print("Cost sensitivity — thresholds chosen on 2022, evaluated on 2023")
    print(results.to_string(index=False))

    output = Path("outputs/sensitivity_analysis.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
