import kagglehub
import shutil
import os
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")

print("Path to dataset files:", path)

# Copy to assignment_2/data folder
target_dir = Path(__file__).parent.parent.parent / "data" / "imdb"
target_dir.mkdir(parents=True, exist_ok=True)

# Copy all files from downloaded path to target directory
for file in Path(path).glob("*"):
    if file.is_file():
        shutil.copy2(file, target_dir / file.name)
        print(f"Copied {file.name} to {target_dir}")

print(f"\nDataset copied to: {target_dir}")