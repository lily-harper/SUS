"""Join game metadata and keep games with enough pre-2024 submissions."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/data/v1"
OUTPUT = ROOT / "data/clean_runs.csv"
MIN_SUBS = 600

sys.path.insert(0, str(ROOT / "src"))
from cleaning import joins, select_games, drop_rows, drop_columns, refine_target

def main():
    runs = pd.read_csv(RAW / "runs-data.csv")
    games = pd.read_csv(RAW / "games-data.csv")
    devs = pd.read_csv(RAW / "developers-data.csv")
    platforms = pd.read_csv(RAW / "platforms-data.csv")
    categories = pd.read_csv(RAW / "categories-data.csv")

    df = joins(games, runs, devs, platforms, categories)

    df = select_games(df, MIN_SUBS=MIN_SUBS)

    df = drop_rows(df)
    df = refine_target(df)
    df = drop_columns(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} runs from {df['gameName'].nunique()} games to {OUTPUT}")


if __name__ == "__main__":
    main()
