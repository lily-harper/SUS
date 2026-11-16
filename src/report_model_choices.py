"""Write a readable summary of saved model choices and hyperparameters."""

import json
from pathlib import Path


def main():
    """Input: saved choice and hyperparameter JSON files. Output: outputs/model_choices.txt."""
    root = Path(__file__).resolve().parent.parent
    choice = json.loads((root / "outputs/selected_model_2023.json").read_text())
    parameter_files = {
        "logreg": root / "src/params/logreg_2021_best_params.json",
        "tree": root / "src/params/tree_2021_best_params.json",
        "forest": root / "src/params/forest_2021_best_params.json",
    }

    lines = [
        "Model selection summary",
        "Selected on: 2023 data",
        f"Winner: {choice['model']}",
        f"Selected threshold: {choice['threshold']}",
        "",
        "2021 tuned hyperparameters:",
    ]

    for name, path in parameter_files.items():
        params = json.loads(path.read_text())
        lines.append(f"{name}{' (winner)' if name == choice['model'] else ''}:")
        for key, value in sorted(params.items()):
            lines.append(f"  {key}: {value}")
        if name == choice["model"]:
            lines.append(f"  Matches frozen choice: {params == choice.get('params')}")
        lines.append("")

    output = root / "outputs/model_choices.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
