import pandas as pd

from paths import PREPROCESSED_RUNS
from src.evaluations import evaluate_baselines


VALIDATION_YEARS = (2021, 2022, 2023)


def load_dated_runs():
    df = pd.read_csv(PREPROCESSED_RUNS, parse_dates=["date"])
    return df.dropna(subset=["date"])


def evaluate_baseline_years(df, years=VALIDATION_YEARS):
    results = []

    for year in years:
        validation = df[df["date"].dt.year == year]
        year_results = evaluate_baselines(validation)
        year_results.insert(0, "year", year)
        results.append(year_results)

    return pd.concat(results, ignore_index=True)


def main():
    df = load_dated_runs()
    print(evaluate_baseline_years(df).to_string(index=False))


if __name__ == "__main__":
    main()
