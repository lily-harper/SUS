"""Show isolation forest anomalies in a two-dimensional PCA projection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

from src.prep import make_preprocessor, make_xy


OUTPUT = Path("outputs")


def main():
    """Input: train, 2022 threshold, and 2023 selection CSVs. Output: a 2023 PCA plot, scores, and cost."""
    train = pd.read_csv("data/train.csv")
    threshold = pd.read_csv("data/threshold.csv")
    selection = pd.read_csv("data/selection.csv")

    X_train, y_train = make_xy(train)
    X_threshold, y_threshold = make_xy(threshold)
    X_selection, y_selection = make_xy(selection)

    # Fit preprocessing, isolation forest, and PCA on accepted training runs only.
    accepted = X_train.loc[y_train == 0]
    preprocessor = make_preprocessor()
    train_features = preprocessor.fit_transform(accepted)
    threshold_features = preprocessor.transform(X_threshold)
    selection_features = preprocessor.transform(X_selection)

    forest = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    forest.fit(train_features)

    # The forest scores the full feature space. PCA only places points on a 2D plot.
    threshold_scores = -forest.score_samples(threshold_features)
    selection_scores = -forest.score_samples(selection_features)
    pca = PCA(n_components=2, svd_solver="arpack", random_state=42)
    pca.fit(train_features)
    points = pca.transform(selection_features)

    # Choose the 2022 cutoff using the project's FP + 2 * FN cost.
    cutoffs = np.unique(np.quantile(threshold_scores, np.linspace(0, 1, 101)))
    actual_threshold = y_threshold.to_numpy()
    costs = []
    for cutoff in cutoffs:
        flagged = threshold_scores >= cutoff
        false_positive = np.sum((actual_threshold == 0) & flagged)
        false_negative = np.sum((actual_threshold == 1) & ~flagged)
        costs.append(false_positive + 2 * false_negative)
    best_cutoff = cutoffs[np.argmin(costs)]

    # Apply the unchanged cutoff to the 2023 model-selection rows.
    flagged = selection_scores >= best_cutoff
    tn, fp, fn, tp = confusion_matrix(y_selection, flagged, labels=[0, 1]).ravel()
    cost = fp + 2 * fn
    print(f"Isolation forest cutoff from 2022: {best_cutoff:.4f}")
    print(f"2023 TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"2023 cost per 1,000: {1000 * cost / len(selection):.2f}")
    print(f"PCA variance shown: {pca.explained_variance_ratio_.sum():.1%}")

    OUTPUT.mkdir(exist_ok=True)
    results = selection[["date", "rejected"]].copy()
    results["pc1"] = points[:, 0]
    results["pc2"] = points[:, 1]
    results["anomaly_score"] = selection_scores
    results["priority_review"] = flagged.astype(int)
    results.to_csv(OUTPUT / "isolation_pca_2023_scores.csv", index=False)

    # Sample points for a readable plot; the scores and metrics above use all rows.
    shown = np.random.default_rng(42).choice(len(selection), size=min(6000, len(selection)), replace=False)
    normal = shown[~flagged[shown]]
    anomalous = shown[flagged[shown]]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(points[normal, 0], points[normal, 1], s=8, alpha=0.2, label="Normal queue")
    ax.scatter(points[anomalous, 0], points[anomalous, 1], s=12, alpha=0.5, label="Priority review")
    ax.set(xlabel="Principal component 1", ylabel="Principal component 2",
           title="2023 isolation forest flags projected with PCA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "isolation_pca_2023.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
