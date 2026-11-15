"""Evaluate a GMM anomaly rule for priority review on 2023 runs."""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture

from src.config import CLEAN_DATA, SEED
from src.prep import make_preprocessor, make_xy, temporal_split


def main():
    """Input: cleaned runs CSV. Output: 2023 review metrics and a threshold CSV."""
    runs = pd.read_csv(CLEAN_DATA)

    train, validation = temporal_split(runs, 2023)
    X_train, _ = make_xy(train)
    X_val, y_val = make_xy(validation)

    # Fit every transformation using training runs only.
    preprocessor = make_preprocessor()
    train_features = preprocessor.fit_transform(X_train)
    val_features = preprocessor.transform(X_val)

    # GMM needs a manageable number of continuous features.
    svd = TruncatedSVD(n_components=20, random_state=SEED)
    train_reduced = svd.fit_transform(train_features)
    val_reduced = svd.transform(val_features)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="diag",
        random_state=SEED,
    )
    gmm.fit(train_reduced)

    # Higher scores mean the GMM considers a run more unusual.
    anomaly_scores = -gmm.score_samples(val_reduced)

    rows = []
    for review_rate in np.arange(0.05, 0.51, 0.05):
        cutoff = np.quantile(anomaly_scores, 1 - review_rate)
        review = anomaly_scores >= cutoff

        false_positive = int(((y_val == 0) & review).sum())
        false_negative = int(((y_val == 1) & ~review).sum())
        true_positive = int(((y_val == 1) & review).sum())
        true_negative = int(((y_val == 0) & ~review).sum())
        cost = false_positive + 2 * false_negative

        rows.append({
            "target_review_rate": review_rate,
            "score_cutoff": cutoff,
            "actual_review_rate": review.mean(),
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
            "cost": cost,
            "cost_per_1000": 1000 * cost / len(y_val),
        })

    results = pd.DataFrame(rows)
    results.to_csv("outputs/thresholds/gmm_threshold.csv", index=False)

    best = results.sort_values(["cost", "actual_review_rate"]).iloc[0]
    review_none_cost = 2 * int((y_val == 1).sum())
    review_all_cost = int((y_val == 0).sum())

    print("GMM anomaly review — 2023 validation")
    print(f"Runs: {len(y_val):,}")
    print(f"Best score cutoff: {best['score_cutoff']:.4f}")
    print(f"Review rate: {best['actual_review_rate']:.1%}")
    print(
        f"Confusion matrix (TN, FP, FN, TP): "
        f"{int(best['true_negative'])}, "
        f"{int(best['false_positive'])}, "
        f"{int(best['false_negative'])}, "
        f"{int(best['true_positive'])}"
    )
    print(f"GMM cost per 1,000: {best['cost_per_1000']:.2f}")
    print(f"Review none per 1,000: {1000 * review_none_cost / len(y_val):.2f}")
    print(f"Review all per 1,000: {1000 * review_all_cost / len(y_val):.2f}")
    print("Saved outputs/thresholds/gmm_threshold.csv")


if __name__ == "__main__":
    main()