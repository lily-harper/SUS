import pandas as pd 

from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def temporal_split(df: pd.DataFrame, validation_year: int):
    dates = pd.to_datetime(df["date"], errors = "coerce")

    start = pd.Timestamp(f"{validation_year}-01-01")
    end   = pd.Timestamp(f"{validation_year + 1}-01-01")

    train = df.loc[dates < start].copy()
    validation = df.loc[(dates >= start) & (dates < end)].copy()

    return train, validation

def make_preprocessor():
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy = "most_frequent")),
        ("encode", OneHotEncoder(handle_unknown = "ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC),
        ("categorical", categorical_steps, CATEGORICAL)
    ])

def make_xy(df: pd.DataFrame):
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df["rejected"].copy()

    return X, y

NUMERIC = [
    "primaryTime",
    "realTime",
    "realTimeNoLoads",
    "inGameTime",
    "numPlayers",
    "releaseYear",
]

CATEGORICAL = [
    "platformName",
    "isEmulated",
    "gameName",
    "developerNames",
    "categoryName",
    "type",
]


