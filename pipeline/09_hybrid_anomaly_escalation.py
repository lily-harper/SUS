"""Compare supervised review rules with an added anomaly escalation rule."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.forest import make_forest
from src.models.logistic import make_log_reg
from src.models.thresholds import choose_thresholds
from src.models.tree import make_tree
from src.models.tuning import PARAMS_PATH_FOREST, PARAMS_PATH_LOG, PARAMS_PATH_TREE
from src.prep import make_preprocessor, make_xy


def counts(y_true, flags):
    """Input: true 0/1 labels and review flags. Output: TN, FP, FN, TP, and 2:1 cost."""
    actual = np.asarray(y_true) == 1
    tn = int((~actual & ~flags).sum())
    fp = int((~actual & flags).sum())
    fn = int((actual & ~flags).sum())
    tp = int((actual & flags).sum())
    return tn, fp, fn, tp, fp + 2 * fn


def main():
    """Input: five year splits, saved parameters, and threshold CSVs. Output: printed results and one CSV."""
    parameter_paths = (PARAMS_PATH_LOG, PARAMS_PATH_TREE, PARAMS_PATH_FOREST)
    for path in parameter_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Saved parameters required; refusing to retune: {path}")

    data = {}
    for name in ("train", "tune", "threshold", "selection", "test"):
        data[name] = make_xy(pd.read_csv(f"data/{name}.csv"))
    X_train, y_train = data["train"]
    X_tune, y_tune = data["tune"]

    names = ("logreg", "tree", "forest")
    tables = {name: pd.read_csv(f"outputs/thresholds/{name}_threshold.csv") for name in names}
    thresholds = json.loads(choose_thresholds(tables))
    missing = set(names) - thresholds.keys()
    if missing:
        raise ValueError(f"No already-selected threshold for: {', '.join(sorted(missing))}")

    # Match pipeline/07: learn normal patterns from verified training rows only.
    accepted = X_train.loc[y_train == 0]
    preprocessor = make_preprocessor()
    train_features = preprocessor.fit_transform(accepted)
    isolation = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    isolation.fit(train_features)
    anomaly_scores = {}
    for split in ("threshold", "selection", "test"):
        X, _ = data[split]
        anomaly_scores[split] = -isolation.score_samples(preprocessor.transform(X))

    models = {"logreg": make_log_reg, "tree": make_tree, "forest": make_forest}
    rows = []
    print("Hybrid anomaly escalation (FP + 2 × FN)")
    print("Model   Split  Rule          TN      FP      FN      TP  Cost/1,000")
    for name, make_model in models.items():
        model = make_model(X_train, y_train, X_tune, y_tune)
        model.fit(X_train, y_train)
        supervised = {}
        for split in ("threshold", "selection", "test"):
            X, _ = data[split]
            supervised[split] = model.predict_proba(X)[:, 1] >= thresholds[name]

        y_2022 = data["threshold"][1]
        candidates = np.quantile(anomaly_scores["threshold"], np.linspace(0.5, 0.999, 50))
        best_cutoff = min(
            candidates,
            key=lambda cutoff: counts(
                y_2022, supervised["threshold"] | (anomaly_scores["threshold"] >= cutoff)
            )[4],
        )

        print(f"\n{name}: supervised threshold {thresholds[name]}, escalation cutoff {best_cutoff:.6f}")
        for split, year in (("threshold", "2022"), ("selection", "2023"), ("test", "2024")):
            _, y_true = data[split]
            alone = supervised[split]
            hybrid = alone | (anomaly_scores[split] >= best_cutoff)
            extra = hybrid & ~alone
            actual = np.asarray(y_true) == 1
            recovered = int((actual & extra).sum())
            added_fp = int((~actual & extra).sum())
            alone_counts = counts(y_true, alone)
            hybrid_counts = counts(y_true, hybrid)
            alone_cost = 1000 * alone_counts[4] / len(y_true)
            hybrid_cost = 1000 * hybrid_counts[4] / len(y_true)

            for rule, result, cost in (("supervised", alone_counts, alone_cost),
                                       ("hybrid", hybrid_counts, hybrid_cost)):
                tn, fp, fn, tp, _ = result
                print(f"{name:7} {year:5} {rule:11} {tn:7} {fp:7} {fn:7} {tp:7} {cost:11.2f}")
            print(f"  {year} escalation adds {recovered} true positives and {added_fp} false positives")

            rows.append({
                "model": name,
                "escalation_cutoff": best_cutoff,
                "split": year,
                "supervised_only_cost_per_1000": alone_cost,
                "hybrid_cost_per_1000": hybrid_cost,
                "recovered_true_positives": recovered,
                "added_false_positives": added_fp,
            })

    output = Path("outputs/hybrid_escalation_analysis.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
