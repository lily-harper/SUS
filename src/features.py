import pandas as pd


def add_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df: run rows containing primaryTime and realTime in seconds.

    Output:
        A copy of df with primary_equals_real_time set to 1 when primaryTime
        and realTime differ by less than 0.01 seconds, otherwise 0.
    """
    result = df.copy()
    difference = (result["primaryTime"] - result["realTime"]).abs()
    result["primary_equals_real_time"] = (difference < 0.01).astype(int)
    return result


def joins(
    gamesDF: pd.DataFrame,
    runsDF: pd.DataFrame,
    devDF: pd.DataFrame,
    platformDF: pd.DataFrame,
    categoryDF: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input:
        gamesDF: game metadata with gameId and developers.
        runsDF: run rows with gameId, categoryId, and platform.
        devDF: developerId and developer name lookup table.
        platformDF: platformId, name, and releaseYear lookup table.
        categoryDF: categoryId and category metadata lookup table.

    Output:
        Run rows joined with game, developer, category, and platform features.
    """
    games = gamesDF[["gameId", "gameName", "developers"]].rename(
        columns={"developers": "developerIds"}
    ).copy()

    names = devDF.set_index("developerId")["name"].to_dict()
    games["developerNames"] = games["developerIds"].map(
        lambda ids: ", ".join(names.get(i, i) for i in ids.split(",") if i)
        if isinstance(ids, str) and ids.strip()
        else pd.NA
    )

    categories = categoryDF[["categoryId", "categoryName", "type", "numPlayers"]]
    platforms = platformDF[["platformId", "name", "releaseYear"]].rename(
        columns={"platformId": "platform", "name": "platformName"}
    )

    runs = runsDF.merge(games, on="gameId", how="left", validate="many_to_one")
    runs = runs.merge(categories, on="categoryId", how="left", validate="many_to_one")
    return runs.merge(platforms, on="platform", how="left", validate="many_to_one")
