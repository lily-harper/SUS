"""Plots for model evaluation."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_rejection_over_time(df):
    """Input: cleaned runs with date and rejected. Output: saved monthly rejection-rate PNG."""
    runs = df.copy()
    runs["month"] = pd.to_datetime(runs["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = runs.groupby("month")["rejected"].mean()
    ax = (monthly * 100).plot(figsize=(11, 4), marker="o", markersize=3)
    ax.axvline(pd.Timestamp("2024-01-01"), color="gray", linestyle="--", label="2024 test starts")
    ax.set(title="Monthly rejection rate", xlabel="Submission month", ylabel="Rejected runs (%)")
    ax.legend()
    figure = ax.figure
    figure.tight_layout()
    output = Path(__file__).resolve().parent.parent / "outputs/reject_over_time.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_covariance(train, numeric_columns):
    """Input: training rows and numeric column names. Output: covariance heatmap."""
    covariance = train[numeric_columns].cov()
    plt.figure(figsize=(8, 6))
    plt.imshow(covariance, cmap="coolwarm")
    plt.colorbar(label="Covariance")
    plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
    plt.yticks(range(len(numeric_columns)), numeric_columns)
    plt.title("Covariance matrix: training data")
    plt.tight_layout()
    plt.show()


def plot_roc_pr(y_true, scores):
    """Input: true 0/1 labels and model names mapped to positive-class scores. Output: two PNG plots."""
    for curve_name, curve_function, x_label, y_label in (
        ("roc", roc_curve, "False positive rate", "True positive rate"),
        ("pr", precision_recall_curve, "Recall", "Precision"),
    ):
        fig, ax = plt.subplots()
        for name, model_scores in scores.items():
            first, second, _ = curve_function(y_true, model_scores)
            x, y = (second, first) if curve_name == "pr" else (first, second)
            label = f"{name} (AUC = {auc(x, y):.3f})" if curve_name == "roc" else name
            ax.plot(x, y, label=label)

        if curve_name == "roc":
            ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
        else:
            ax.axhline(y_true.mean(), color="black", linestyle="--", label="Class prevalence")

        ax.set(xlabel=x_label, ylabel=y_label, title=f"2023 validation {curve_name.upper()} curve")
        ax.legend(loc="lower right" if curve_name == "roc" else "best")
        fig.tight_layout()
        fig.savefig(f"outputs/{curve_name}_2023.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    cleaned_runs = pd.read_csv(root / "data/clean_runs.csv")
    plot_rejection_over_time(cleaned_runs)
