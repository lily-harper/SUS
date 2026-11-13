"""Download raw data and produce the cleaned runs CSV."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/data/v1"
REQUIRED = ("runs-data.csv", "games-data.csv", "developers-data.csv")


def main():
    if not all((RAW / name).is_file() for name in REQUIRED):
        print("Running 01_get_kaggle_data.py", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "pipeline/01_get_kaggle_data.py")], check=True
        )
    else:
        print("Raw Kaggle data found; skipping 01_get_kaggle_data.py", flush=True)

    print("Running 02_clean_runs.py", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/02_clean_runs.py")], check=True)


if __name__ == "__main__":
    main()
