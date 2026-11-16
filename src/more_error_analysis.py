import pandas as pd
train = pd.read_csv("data/train.csv")
clean = pd.read_csv("data/clean_runs.csv")

print("GTA Category Extensions rows in 2018-2020 train:",
      (train["gameName"] == "Grand Theft Auto Category Extensions").sum())
print(clean.loc[clean["gameName"] == "Grand Theft Auto Category Extensions",
                 "categoryName"].value_counts())
print(clean.loc[clean["developerNames"].str.contains("Jukereise", na=False),
                 ["gameName", "platformName", "type"]].drop_duplicates())

test = pd.read_csv("data/test.csv")

gta = test.loc[test["gameName"] == "Grand Theft Auto Category Extensions"]
print(gta.groupby("categoryName")["rejected"].agg(["size", "mean"]))

juke = test.loc[test["developerNames"].str.contains("Jukereise", na=False)]
print(juke.groupby(["platformName", "type"])["rejected"].agg(["size", "mean"]))