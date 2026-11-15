from pathlib import Path

import pandas as pd

from src.config import CLEAN_DATA
from src.models.logistic import candidates, make_logistic_pipeline, thresholds
from src.prep import make_xy, temporal_split
from models.thresholds import threshold_metrics


df = pd.read_csv(CLEAN_DATA)
train, validation = temporal_split(df, 2022)
X_train, y_train = make_xy(train)
X_validation, y_validation = make_xy(validation)

results = []

for candidate in candidates:
    model = make_logistic_pipeline(
        C=candidate["C"],
        class_weight=candidate["class_weight"],
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_validation)[:, 1]

    candidate_results = threshold_metrics(
        y_true=y_validation,
        probabilities=probabilities,
        thresholds=thresholds,
        model_name="logistic",
        year=2022,
    )
    candidate_results["C"] = candidate["C"]
    candidate_results["class_weight"] = candidate["class_weight"]
    results.append(candidate_results)

Path("outputs").mkdir(parents=True, exist_ok=True)
pd.concat(results, ignore_index=True).to_csv(
    "outputs/logistic_2022_candidates.csv",
    index=False,
)
