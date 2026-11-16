import pandas as pd

th = pd.read_csv("data/threshold.csv")
n = len(th)
positives = (th["rejected"] == 1).sum()

for name in ["logreg", "tree", "forest"]:
    t = pd.read_csv(f"outputs/thresholds/{name}_threshold.csv")
    t["false_negative"] = (t["cost"] - t["false_positive"]) / 2
    for m in [2, 5, 10]:
        t[f"cost_per_1000_{m}"] = 1000 * (t["false_positive"] + m * t["false_negative"]) / n
    print(name, "\n", t[["threshold", f"cost_per_1000_2", f"cost_per_1000_5", f"cost_per_1000_10"]])

for m in [2, 5, 10]:
    print(f"review-none at {m}:1 =", 1000 * (m * positives) / n)