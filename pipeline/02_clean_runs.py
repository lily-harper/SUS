"""Join game metadata and keep games with enough pre-2024 submissions."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/data/v1"
OUTPUT = ROOT / "data/clean_runs.csv"
MIN_SUBS = 1250
MAX_REVIEWERS = 20

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from cleaning import clean_release_year, select_games, drop_rows, drop_columns, refine_target
from src.features import add_timing_features, joins
from src.prep import save_year_splits

def main():
    runs = pd.read_csv(RAW / "runs-data.csv")
    games = pd.read_csv(RAW / "games-data.csv")
    devs = pd.read_csv(RAW / "developers-data.csv")
    platforms = pd.read_csv(RAW / "platforms-data.csv")
    categories = pd.read_csv(RAW / "categories-data.csv")

    df = drop_rows(runs)
    df = joins(games, df, devs, platforms, categories)
    df = clean_release_year(df)

    dates = pd.to_datetime(df["date"], errors="coerce")
    candidate_pool = df.loc[dates < pd.Timestamp("2021-01-01")]
    candidates = candidate_pool.groupby(["gameId", "gameName"]).agg(
        submissions=("runId", "size"),
        reviewers=("examiner", "nunique"),
    ).reset_index()
    candidates = candidates.loc[
        (candidates["submissions"] > MIN_SUBS)
        & (candidates["reviewers"] < MAX_REVIEWERS)
    ].sort_values("submissions", ascending=False)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    candidates.to_csv(ROOT / "outputs/game_queue_candidates.csv", index=False)

    df = select_games(
        df,
        MIN_SUBS=MIN_SUBS,
        MAX_REVIEWERS=MAX_REVIEWERS,
    )
    df = add_timing_features(df)
    df = refine_target(df)
    df = drop_columns(df)

    df = df.drop(columns = ["gameId", "categoryId", "levelId", "developerIds", "platform"])
    df = df[pd.to_datetime(df["date"]).dt.year >= 2018].copy()  

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} runs from {df['gameName'].nunique()} games to {OUTPUT}")

    save_year_splits(df)


if __name__ == "__main__":
    main()
