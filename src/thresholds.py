import pandas as pd 

def threshold_metrics(y_true, probabilities, thresholds, model_name, year):
    y_true = pd.Series(y_true).reset_index(drop=True)
    probabilities = pd.Series(probabilities).reset_index(drop=True)

    rows = []

    for threshold in thresholds:
        review = probabilities >= threshold

        tp = ((y_true == 1) & review).sum()
        fp = ((y_true == 0) & review).sum()
        tn = ((y_true == 0) & ~review).sum()
        fn = ((y_true == 1) & ~review).sum()

        total = len(y_true)
        reviewed = tp + fp
        cost = fp + 2 * fn

        precision = (
            tp / reviewed
            if reviewed > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        rows.append({
            "model": model_name,
            "validation_year": year,
            "threshold": threshold,
            "total_runs": total,
            "reviewed_runs": reviewed,
            "review_rate": reviewed / total,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "cost": cost,
            "cost_per_1000": 1000 * cost / total,
            "precision": precision,
            "rejection_recall": recall,
            "F1": f1,
        })

    return pd.DataFrame(rows)
