from pathlib import Path

import pandas as pd

from src.config import CLEAN_DATA
from src.models import make_logistic_pipeline, make_random_forest_pipeline
from src.prep import make_xy, temporal_split
from src.thresholds import threshold_metrics


df = pd.read_csv(CLEAN_DATA)
pre_2024, holdout_2024 = temporal_split(df, 2024)

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
model_builders = {
    "logistic": make_logistic_pipeline,
    "random_forest": make_random_forest_pipeline,
}

all_model_results = []

for model_name, make_model in model_builders.items():
    for year in (2021, 2022, 2023):
        train, validation = temporal_split(pre_2024, year)
        X_train, y_train = make_xy(train)
        X_validation, y_validation = make_xy(validation)

        model = make_model()
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]

        all_model_results.append(
            threshold_metrics(
                y_true=y_validation,
                probabilities=probabilities,
                thresholds=thresholds,
                model_name=model_name,
                year=year,
            )
        )

all_results = pd.concat(all_model_results, ignore_index=True)
summary = (
    all_results.groupby(["model", "threshold"], as_index=False)["cost_per_1000"]
    .mean()
    .sort_values("cost_per_1000")
)
chosen = summary.drop_duplicates("model")
chosen_thresholds = dict(zip(chosen["model"], chosen["threshold"]))

X_train, y_train = make_xy(pre_2024)
X_holdout, y_holdout = make_xy(holdout_2024)
final_results = []

for model_name, make_model in model_builders.items():
    model = make_model()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_holdout)[:, 1]

    final_results.append(
        threshold_metrics(
            y_true=y_holdout,
            probabilities=probabilities,
            thresholds=[chosen_thresholds[model_name]],
            model_name=model_name,
            year=2024,
        )
    )

Path("outputs").mkdir(parents=True, exist_ok=True)
all_results.to_csv("outputs/model_threshold_results.csv", index=False)
summary.to_csv("outputs/model_threshold_summary.csv", index=False)
pd.concat(final_results, ignore_index=True).to_csv(
    "outputs/final_model_results.csv",
    index=False,
)
print(summary.to_string(index=False))
