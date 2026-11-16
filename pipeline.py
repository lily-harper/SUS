"""Download raw data and produce the cleaned runs CSV."""

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/data/v1"

REQUIRED = (
    "runs-data.csv",
    "games-data.csv",
    "developers-data.csv",
    "platforms-data.csv",
    "categories-data.csv",
)


def main():
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing_path}" if existing_path else str(ROOT)

    print("\n\tRunning 01_get_kaggle_data.py")
    if not all((RAW / name).is_file() for name in REQUIRED):
        print("Pulling data from Kaggle", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "pipeline/01_get_kaggle_data.py")], check=True, cwd=ROOT, env=environment
        )
    else:
        print("Raw Kaggle data found; skipping 01_get_kaggle_data.py", flush=True)

    print("\n\tRunning 02_clean_runs.py\n", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/02_clean_runs.py")], check=True, cwd=ROOT, env=environment)

    print("\n\tRunning 03_run_models.py\n", flush=True)
    (ROOT / "outputs/thresholds").mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/03_run_models.py")], check=True, cwd=ROOT, env=environment)

    print("\n\tRunning 05_test_2024.py\n", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/05_test_2024.py")], check=True, cwd=ROOT, env=environment)

    print("\n\tRunning 07_isolation_pca_2d.py\n", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/07_isolation_pca_2d.py")], check=True, cwd=ROOT, env=environment)

    print("\n\tRunning 08_cost_sensitivity.py\n", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/08_cost_sensitivity.py")], check=True, cwd=ROOT, env=environment)

    print("\n\tRunning 09_error_analysis.py\n", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/09_error_analysis.py")], check=True, cwd=ROOT, env=environment)

    print("\n\tRunning 10_hybrid_anomaly_escalation.py\n", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline/10_hybrid_anomaly_escalation.py")], check=True, cwd=ROOT, env=environment)
    

if __name__ == "__main__":
    main()
