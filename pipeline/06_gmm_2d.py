"""Plot a two-feature probabilistic anomaly model without using run times."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURES = ["releaseYear", "numPlayers"]
OUTPUT = Path("outputs")


def main():
    """Input: 2018–2020 train, 2022 threshold, and 2023 selection CSVs. Output: a 2023 anomaly plot and metrics."""
    train = pd.read_csv("data/train.csv")
    threshold = pd.read_csv("data/threshold.csv")
    selection = pd.read_csv("data/selection.csv")

    # Learn missing-value replacements and axis scales from accepted training runs.
    accepted = train.loc[train["rejected"] == 0, FEATURES]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_points = scaler.fit_transform(imputer.fit_transform(accepted))
    threshold_points = scaler.transform(imputer.transform(threshold[FEATURES]))
    selection_points = scaler.transform(imputer.transform(selection[FEATURES]))

    # Each Gaussian describes one region of ordinary runs in the two-dimensional plane.
    model = GaussianMixture(n_components=3, covariance_type="full", reg_covar=0.01, random_state=67)
    model.fit(train_points)

    # A lower fitted density gives a higher anomaly score. It is not a rejection probability.
    threshold_scores = -model.score_samples(threshold_points)
    selection_scores = -model.score_samples(selection_points)

    # Select the cutoff using 2022 labels and the project's FP + 2 * FN cost.
    cutoffs = np.unique(np.quantile(threshold_scores, np.linspace(0, 1, 101)))
    y_threshold = threshold["rejected"].to_numpy()
    costs = []
    for cutoff in cutoffs:
        predictions = threshold_scores >= cutoff
        false_positive = np.sum((y_threshold == 0) & predictions)
        false_negative = np.sum((y_threshold == 1) & ~predictions)
        costs.append(false_positive + 2 * false_negative)
    best_cutoff = cutoffs[np.argmin(costs)]

    # Apply the unchanged 2022 cutoff to the 2023 model-selection rows.
    predictions = selection_scores >= best_cutoff
    y_selection = selection["rejected"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_selection, predictions, labels=[0, 1]).ravel()
    cost = fp + 2 * fn
    print(f"Two-feature GMM cutoff from 2022: {best_cutoff:.3f}")
    print(f"2023 TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"2023 cost per 1,000: {1000 * cost / len(selection):.2f}")

    OUTPUT.mkdir(exist_ok=True)
    results = selection[["date", *FEATURES, "rejected"]].copy()
    results["anomaly_score"] = selection_scores
    results["priority_review"] = predictions.astype(int)
    results.to_csv(OUTPUT / "gmm_2d_2023_scores.csv", index=False)

    # Sample only for legibility; the model, cutoff, and metrics use all rows.
    shown = np.random.default_rng(67).choice(len(selection), size=min(5000, len(selection)), replace=False)
    x = selection_points[shown, 0]
    y = selection_points[shown, 1]
    shown_anomalies = predictions[shown]

    x_grid = np.linspace(np.percentile(x, 1) - 0.5, np.percentile(x, 99) + 0.5, 200)
    y_grid = np.linspace(np.percentile(y, 1) - 0.5, np.percentile(y, 99) + 0.5, 200)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_scores = -model.score_samples(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(x[~shown_anomalies], y[~shown_anomalies], s=9, alpha=0.15, label="Normal queue")
    ax.scatter(x[shown_anomalies], y[shown_anomalies], s=12, alpha=0.35, label="Priority review")
    ax.contour(xx, yy, grid_scores, levels=[best_cutoff], colors="black", linewidths=1.5)
    ax.set(xlabel="Release year (standardized)", ylabel="Number of players (standardized)",
           title="2023 runs: two-feature Gaussian mixture anomaly rule")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "gmm_2d_2023.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
