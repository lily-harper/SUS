from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

import pandas as pd

NUMERIC_COLUMNS = ["primaryTime", "realTime", "realTimeNoLoads", "inGameTime"]
CATEGORICAL_COLUMNS = ["gameName", "categoryId", "isEmulated"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

def prep(df: pd.DataFrame) -> pd.Dataframe:
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("string").fillna("missing").astype(str)

    df = df.sort_values("date")

    return df

def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),

        ]), NUMERIC_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])
