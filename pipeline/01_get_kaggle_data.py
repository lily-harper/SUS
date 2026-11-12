import kagglehub
import sys

sys.path.insert(0, ".")
from paths import RAW_DIR


def download_kaggle_data():
    return kagglehub.dataset_download(
        "alexmerren1/speedrun-com-data",
        output_dir=RAW_DIR,
    )

def main():
    path = download_kaggle_data()
    print("Path to dataset files:", path)

if __name__ == "__main__":
    main()
