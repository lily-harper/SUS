import pandas as pd 
from src.config import CLEAN_DATA

df = pd.read_csv(CLEAN_DATA)

df.head()

from src.prep import temporal_split 

train, test = temporal_split(df, 2024)

train[train["rejected"] == 1].shape[0] / len(train)

test[test["rejected"] == 1].shape[0] / len(test)