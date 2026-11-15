from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from src.prep import make_preprocessor 

import argparse

import pandas as pd 

from src.config import data_stuff 
from src.config import TRAIN_DATA

from src.models.logistic import make_log_reg
from src.models.tree import make_tree

def print_things(y_val, y_pred, model, threshold):
    matrix = confusion_matrix(y_val, y_pred, labels = [0, 1])

    print(f"{model} — 2022 validation, threshold {threshold}")
    print("                 Predicted 0  Predicted 1")
    print(f"Actual 0 (TN/FP) {matrix[0, 0]:11}  {matrix[0, 1]:11}")
    print(f"Actual 1 (FN/TP) {matrix[1, 0]:11}  {matrix[1, 1]:11}")

    cost = matrix[0, 1] + 2 * matrix[1, 0]  # FP + 2 × FN
    print(f"Cost: {cost} ({1000 * cost / len(y_val):.2f} per 1,000 runs)")

def lr(threshold: float):
    X_train, y_train, X_val, y_val = data_stuff(TRAIN_DATA)

    model = make_log_reg(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[: ,1]
    y_pred = (probs >= threshold).astype(int) 

    print_things(y_val, y_pred, "tree", threshold)

def tree(threshold: float):
    X_train, y_train, X_val, y_val = data_stuff(TRAIN_DATA)

    model = make_tree(X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[: ,1]
    y_pred = (probs >= threshold).astype(int) 

    print_things(y_val, y_pred, "tree", threshold)

def main():
    print("running logistic regression")
    lr(args.logthreshold)
    tree(args.treethreshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logthreshold", type = float, required = True,
        help = "Logistic regression threshold"
    )
    parser.add_argument(
        "--treethreshold", type = float, required = True,
        help = "tree threshold"
    )

    args = parser.parse_args()

    if not 0 <= args.logthreshold <= 1:
        parser.error("--threhold must be tbetween 0 and 1")

    main()
