"""Download the Speedrun.com Kaggle dataset when the raw CSVs are missing."""

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/data/v1"
DATASET = "alexmerren1/speedrun-com-data"
REQUIRED = ("runs-data.csv", "games-data.csv", "developers-data.csv")


def main():
    if all((RAW / name).is_file() for name in REQUIRED):
        print(f"Raw data already present at {RAW}")
        return

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("Install the Kaggle CLI and configure Kaggle credentials first.")

    destination = ROOT / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [kaggle, "datasets", "download", "-d", DATASET, "-p", str(destination), "--unzip"],
        check=True,
    )
    missing = [name for name in REQUIRED if not (RAW / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Kaggle download did not provide {missing} under {RAW}")
    print(f"Downloaded raw data to {RAW}")


if __name__ == "__main__":
    main()
