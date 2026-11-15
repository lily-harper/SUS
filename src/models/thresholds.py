import pandas as pd 
import matplotlib.pyplot as plt 

def threshold_metrics(y_true, probabilities, save_path = None):
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
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
        })

    thresh = pd.DataFrame(rows)
    thresh.sort_values(by = "cost", ascending = True)

    if save_path:
        thresh.to_csv(f"outputs/thresholds/{save_path}_threshold.csv", index=False, header=True)

    return thresh

def plot_thresholds(thresh: pd.DataFrame, model_name):
    plt.plot(thresh["threshold"], thresh["cost_per_1000"], marker = "o")
    plt.xlabel("Review threshold")
    plt.ylabel("Cost per 1,000 runs")
    plt.title(f"Validation cost by review threshold for {model_name}")
    plt.grid()
    plt.show()
    