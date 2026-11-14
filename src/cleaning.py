from pathlib import Path
import pandas as pd 

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/data/v1"
OUTPUT = ROOT / "data/clean_runs.csv"

def joins(gamesDF: pd.DataFrame, runsDF: pd.DataFrame,
          devDF: pd.DataFrame, platformDF: pd.DataFrame,
          categoryDF: pd.DataFrame) -> pd.DataFrame:
    games = gamesDF[["gameId", "gameName", "developers"]].rename(
        columns={"developers": "developerIds"}
    ).copy()
    names = devDF.set_index("developerId")["name"].to_dict()
    games["developerNames"] = games["developerIds"].map(
        lambda ids: ", ".join(names.get(i, i) for i in ids.split(",") if i)
        if isinstance(ids, str) and ids.strip() else pd.NA
    )
    categories = categoryDF[["categoryId", "categoryName", "type", "numPlayers"]]
    platforms = platformDF[["platformId", "name", "releaseYear"]].rename(
        columns={"platformId": "platform", "name": "platformName"}
    )
    runs = runsDF.merge(games, on="gameId", how="left", validate="many_to_one")
    runs = runs.merge(categories, on="categoryId", how="left", validate="many_to_one")
    return runs.merge(platforms, on="platform", how="left", validate="many_to_one")

def select_games(df: pd.DataFrame, MIN_SUBS: int) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"], errors="coerce")
    counts = df.loc[dates < pd.Timestamp("2024-01-01")].groupby("gameId").size()
    selected_ids = counts.index[counts >= MIN_SUBS]
    return df.loc[
        df["gameId"].isin(selected_ids) & (dates < pd.Timestamp("2025-01-01"))
    ].copy()

# removing things 

def drop_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["runId", "gameId", "date", "status"]).drop_duplicates(
        subset="runId", keep="first"
    ).copy()

def drop_columns(df: pd.DataFrame) -> pd.DataFrame :
    return df.drop(columns=["statusReason", "verifiedDate", "examiner",
                            "variablesAndValues"], errors="ignore")

def refine_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df["status"].isin(["verified", "rejected"])].copy()
    df["rejected"] = df["status"].eq("rejected").astype(int)
    return df.drop(columns="status")

