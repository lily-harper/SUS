import pandas as pd

from src.config import CLEAN_DATA
from src.prep import temporal_split


df = pd.read_csv(CLEAN_DATA)
pre_2024, holdout_2024 = temporal_split(df, 2024)

print("pre-2024 rows:", len(pre_2024))
print("2024 holdout rows:", len(holdout_2024))

for year in (2021, 2022, 2023):
    train, validation = temporal_split(pre_2024, year)
    print(year, "training rows:", len(train), "validation rows:", len(validation))
