"""
Download hotel booking dataset from kaggle into `data/raw/`
"""

import shutil
from pathlib import Path

import kagglehub

DATASET = "youssefaboelwafa/hotel-booking-cancellation-prediction"

# Set project root (two levels up from Scripts)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT/"data"/"raw"

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to cache: {cache_path}")

    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cache_path}")

    for src in csv_files:
        dest = RAW_DIR/src.name
        shutil.copy2(src, dest)
        print(f"Copied {src.name} -> {dest}")

if __name__ == '__main__':
    main()