import sys

import pandas as pd

sys.path.insert(0, ".")
from paths import CLEAN_RUNS, PREPROCESSED_RUNS


columns = [
    "runId", "gameName", "categoryId", "date",
    "primaryTime", "realTime", "realTimeNoLoads", "inGameTime",
    "isEmulated", "players", "variablesAndValues",
    "developerNames", "status",
]

df = pd.read_csv(CLEAN_RUNS, usecols=columns)
df = df[df.status.isin(["verified", "rejected"])].copy()

df["rejected"] = (df.status == "rejected").astype(int)
df["date"] = pd.to_datetime(df.date, errors="coerce")
df = df.drop(columns="status")

print("Runs for modeling:", len(df))
print("Rejected:", df.rejected.sum())
print("Validated:", (df.rejected == 0).sum())
print("prevalence: ", (df["rejected"].mean() * 100))
print("Missing dates:", df.date.isna().sum())

df.to_csv(PREPROCESSED_RUNS, index=False)

