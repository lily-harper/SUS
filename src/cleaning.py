import pandas as pd 

def select_games(
    df: pd.DataFrame,
    MIN_SUBS: int,
    MAX_REVIEWERS: int,
) -> pd.DataFrame:
    """
    Input:
        df: run rows containing gameId and date.
        MIN_SUBS: minimum number of pre-2024 submissions per game.
        MAX_REVIEWERS: exclusive upper bound for unique pre-2024 examiners.

    Output:
        Runs through 2024 for games meeting the submission threshold.
    """
    dates = pd.to_datetime(df["date"], errors="coerce")
    pool = df.loc[dates < pd.Timestamp("2024-01-01")]
    summary = pool.groupby("gameId").agg(
        submissions=("runId", "size"),
        reviewers=("examiner", "nunique"),
    )
    selected_ids = summary.index[
        (summary["submissions"] > MIN_SUBS)
        & (summary["reviewers"] < MAX_REVIEWERS)
    ]
    return df.loc[
        df["gameId"].isin(selected_ids) & (dates < pd.Timestamp("2025-01-01"))
    ].copy()

# removing things 

def drop_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df: raw run rows containing IDs, date, and status.

    Output:
        Rows with required fields, unique run IDs, and dates from 2018 through 2024.
    """
    df = df.dropna(subset=["runId", "gameId", "date", "status"]).drop_duplicates(
        subset="runId", keep="first").copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[
        (dates >= pd.Timestamp("2018-01-01"))
        & (dates < pd.Timestamp("2025-01-01"))
    ].copy()

    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame :
    """Remove review-time and excluded run-setting columns."""
    return df.drop(columns=["statusReason", "verifiedDate", "examiner",
                            "variablesAndValues"], errors="ignore")


def clean_release_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df: run rows containing releaseYear.

    Output:
        A copy of df where release years before 1950 or after 2025 are missing.
    """
    result = df.copy()
    invalid_year = (result["releaseYear"] < 1950) | (result["releaseYear"] > 2025)
    result.loc[invalid_year, "releaseYear"] = pd.NA
    return result


def refine_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df: rows with a status column.

    Output:
        Verified and rejected rows with binary rejected target.
    """
    df = df.loc[df["status"].isin(["verified", "rejected"])].copy()
    df["rejected"] = df["status"].eq("rejected").astype(int)
    return df.drop(columns="status")

