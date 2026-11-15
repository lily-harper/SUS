from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.models.thresholds import save_model_threshold_csvs

import pandas as pd 
from src.prep import temporal_split, make_xy
from src.config import data_stuff 
from src.config import CLEAN_DATA

from src.models.logistic import make_log_reg
from src.models.tree import make_tree
from src.models.dummy import make_dummy
from src.models.forest import make_forest

import json
from src.models.thresholds import choose_thresholds, threshold_metrics

def get_thresholds():
    """Input: 2022 threshold CSVs. Output: model names mapped to thresholds."""
    tables = {
        "logreg": pd.read_csv("outputs/thresholds/logreg_threshold.csv"),
        "tree": pd.read_csv("outputs/thresholds/tree_threshold.csv"),
        "forest": pd.read_csv("outputs/thresholds/forest_threshold.csv"),
    }
    return json.loads(choose_thresholds(tables))

def print_things(y_val, y_pred, model, threshold, ax):
    matrix = confusion_matrix(y_val, y_pred, labels = [0, 1])

    print(f"{model} — 2022 validation, threshold {threshold}")
    print("                 Predicted 0  Predicted 1")
    print(f"Actual 0 (TN/FP) {matrix[0, 0]:11}  {matrix[0, 1]:11}")
    print(f"Actual 1 (FN/TP) {matrix[1, 0]:11}  {matrix[1, 1]:11}")

    cost = matrix[0, 1] + 2 * matrix[1, 0]  # FP + 2 × FN

    display = ConfusionMatrixDisplay(matrix, display_labels=[0, 1])
    display.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model} — threshold {threshold}")
    ax.set_xlabel(f"Predicted label\nCost: {cost} ({1000 * cost / len(y_val):.2f} per 1,000)")

    print(f"Cost: {cost} ({1000 * cost / len(y_val):.2f} per 1,000 runs)")

def lr(threshold, X_train, y_train, X_val, y_val, ax):
    model = make_log_reg(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[: ,1]
    y_pred = (probs >= threshold).astype(int) 

    print_things(y_val, y_pred, "logistic regression", threshold, ax)

def dummy(X_train, y_train, X_val, y_val, ax):
    """Input: probability threshold. Output: dummy validation confusion matrix and cost."""
    df = pd.read_csv(CLEAN_DATA)
    train, val = temporal_split(df, 2023)

    X_train, y_train = make_xy(train)
    X_val, y_val = make_xy(val)

    model = make_dummy()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print_things(y_val, y_pred, "dummy", " ", ax)

def tree(threshold, X_train, y_train, X_val, y_val, ax):

    model = make_tree(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[: ,1]
    y_pred = (probs >= threshold).astype(int) 

    print_things(y_val, y_pred, "tree", threshold, ax)

def forest(threshold: float, X_train, y_train, X_val, y_val, ax):
    """Input: probability threshold. Output: forest validation confusion matrix and cost."""

    model = make_forest(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    print_things(y_val, y_pred, "random forest", threshold, ax)

def main():
    save_model_threshold_csvs()
    X_train, y_train, X_val, y_val = data_stuff(CLEAN_DATA, 2023)
    thresholds = get_thresholds()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    dummy(X_train, y_train, X_val, y_val, axes[0, 0])
    if "logreg" in thresholds:
        lr(thresholds["logreg"], X_train, y_train, X_val, y_val, axes[0, 1])
        print("logistic reg")

    if "tree" in thresholds:
        tree(thresholds["tree"], X_train, y_train, X_val, y_val, axes[1, 0])
        print("tree")

    if "forest" in thresholds:
        forest(thresholds["forest"], X_train, y_train, X_val, y_val, axes[1, 1])
        print("forest")

    fig.tight_layout()
    fig.savefig("outputs/confusion_matrices_2023.png", dpi=200)
    plt.close(fig)

