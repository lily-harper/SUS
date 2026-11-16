"""Save row counts and rejection prevalence for the five year splits."""

from pathlib import Path

import pandas as pd


def main():
    """Input: five data split CSVs. Output: outputs/split_summary.csv."""
    split_names = ["train", "tune", "threshold", "selection", "test"]
    rows = []

    for name in split_names:
        data = pd.read_csv(f"data/{name}.csv", usecols=["rejected"])
        rows.append({
            "dataset": name,
            "row_count": len(data),
            "rejection_prevalence": data["rejected"].mean(),
        })

    output = Path("outputs/split_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
