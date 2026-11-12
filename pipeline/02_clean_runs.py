import pandas as pd
import sys

sys.path.insert(0, ".")
from candidates import CANDIDATES
from paths import RUNS_DATA, GAMES_DATA, DEVELOPERS_DATA, CLEAN_RUNS


runs = pd.read_csv(RUNS_DATA)
games = pd.read_csv(GAMES_DATA, usecols=["gameId", "gameName", "developers"])
developers = pd.read_csv(DEVELOPERS_DATA)

developer_names = dict(zip(developers.developerId, developers.name))


def get_developer_names(ids):
    if pd.isna(ids):
        return None
    return ", ".join(developer_names[developer_id] for developer_id in ids.split(","))


games["developerNames"] = games.developers.apply(get_developer_names)
games = games.rename(columns={"developers": "developerIds"})

print("Runs before cleaning:", len(runs))
runs = runs.drop_duplicates(subset="runId", keep="first")
runs = runs.drop(columns=["levelId", "platform"])
runs = runs.merge(games, on="gameId", how="left")
runs = runs[runs.gameName.isin(CANDIDATES)]
runs = runs[~runs.date.str.startswith("2025", na=False)]

print("Runs after cleaning:", len(runs))

runs.to_csv(CLEAN_RUNS, index=False)
