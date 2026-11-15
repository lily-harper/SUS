import pandas as pd 
import matplotlib.pyplot as plt 
import json
from sklearn.metrics import f1_score
from src.prep import make_xy

def save_model_threshold_csvs():
    """Input: data/train.csv. Output: one 2022 threshold CSV for each trained model."""
    from src.models.logistic import make_log_reg
    from src.models.tree import make_tree
    from src.models.forest import make_forest

    X_train, y_train = make_xy(pd.read_csv("data/train.csv"))
    X_tune, y_tune = make_xy(pd.read_csv("data/tune.csv"))
    X_threshold, y_threshold = make_xy(pd.read_csv("data/threshold.csv"))   

    models = {
        "logreg": make_log_reg,
        "tree": make_tree,
        "forest": make_forest,
    }

    for model_name, make_model in models.items():
        model = make_model(X_train, y_train, X_tune, y_tune)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_threshold)[:, 1]
        threshold_metrics(y_threshold, probabilities, save_path=model_name)

def threshold_metrics(y_true, probabilities, save_path = None):
    thresholds = [0.2, 0.3, 0.4, 0.45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.85, 0.9]
    rows = []

    for threshold in thresholds:
        decision = (probabilities >= threshold).astype(int)
        false_positive = ((y_true == 0) & (decision == 1)).sum()
        false_negative = ((y_true == 1) & (decision == 0)).sum()
        true_negative = ((y_true == 0) & (decision == 0)).sum()
        cost = false_positive + 2 * false_negative

        rows.append({
            "threshold": threshold,
            "false_positive": false_positive, 
            "fpr": false_positive / (false_positive + true_negative),
            "cost": cost,
            "cost_per_1000": 1000 * cost / len(y_true),
            "f1": f1_score(y_true, decision, zero_division=0),
        })

    thresh = pd.DataFrame(rows)
    thresh.sort_values(by = "cost", ascending = True)

    if save_path:
        thresh.to_csv(f"outputs/thresholds/{save_path}_threshold.csv", index=False, header=True)

    return thresh

def choose_thresholds(model_results):
    """Input: model names mapped to threshold tables. Output: JSON with each eligible model's best threshold."""
    chosen = {}
    for model_name, table in model_results.items():
        eligible = table.loc[table["f1"] > 0.1]
        if not eligible.empty:
            best = eligible.sort_values(["cost", "threshold"]).iloc[0]
            chosen[model_name] = float(best["threshold"])
    return json.dumps(chosen, indent=2)

def plot_thresholds(thresh: pd.DataFrame, model_name):
    plt.plot(thresh["threshold"], thresh["cost_per_1000"], marker = "o")
    plt.xlabel("Review threshold")
    plt.ylabel("Cost per 1,000 runs")
    plt.title(f"Validation cost by review threshold for {model_name}")
    plt.grid()
    plt.show()
    
