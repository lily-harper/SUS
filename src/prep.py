import pandas as pd 
import numpy as np

from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def save_year_splits(df: pd.DataFrame) -> None:
    """
    Input: cleaned runs with a date column.
    Output: five CSV files in data/, one for each stage.
    """
    dates = pd.to_datetime(df["date"], errors="coerce")

    splits = {
        "data/train.csv": dates < "2021-01-01",
        "data/tune.csv": (dates >= "2021-01-01") & (dates < "2022-01-01"),
        "data/threshold.csv": (dates >= "2022-01-01") & (dates < "2023-01-01"),
        "data/selection.csv": (dates >= "2023-01-01") & (dates < "2024-01-01"),
        "data/test.csv": (dates >= "2024-01-01") & (dates < "2025-01-01"),
    }

    for path, rows in splits.items():
        split = df.loc[rows].copy()
        split.to_csv(path, index=False)
        print(f"Saved {len(split):,} runs to {path}")

def temporal_split(df: pd.DataFrame, validation_year: int):
    dates = pd.to_datetime(df["date"], errors = "coerce")

    start = pd.Timestamp(f"{validation_year}-01-01")
    end   = pd.Timestamp(f"{validation_year + 1}-01-01")

    train = df.loc[dates < start].copy()
    validation = df.loc[(dates >= start) & (dates < end)].copy()

    return train, validation

def save_temporal_split(df: pd.DataFrame, year) -> None:
    """
    Input:
        df: cleaned runs containing a date column.

    Output:
        Writes runs before 2023 to data/train.csv and 2023 runs to data/test.csv.
    """
    train, test = temporal_split(df, year)

    train.to_csv("data/train.csv", index = False)
    print(f"Saved {len(train):,} training runs games to data/train.csv")

    test.to_csv("data/test.csv", index = False)
    print(f"Saved {len(test):,} test runs from games to data/test.csv")


def make_preprocessor(): 
    time_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("times", time_steps, TIME_COLUMNS),
        ("numeric", numeric_steps, OTHER_NUMERIC),
        ("categorical", categorical_steps, CATEGORICAL),
    ])

def make_xy(df: pd.DataFrame):
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df["rejected"].copy()

    return X, y

TIME_COLUMNS = [
    "primaryTime",
    "realTimeNoLoads",
    "inGameTime",
]

OTHER_NUMERIC = [
    "primary_equals_real_time",
    "numPlayers",
    "releaseYear",
]

CATEGORICAL = [
    "platformName",
    "isEmulated",
    "gameName",
    "categoryName",
    "type",
]

NUMERIC = TIME_COLUMNS + OTHER_NUMERIC
