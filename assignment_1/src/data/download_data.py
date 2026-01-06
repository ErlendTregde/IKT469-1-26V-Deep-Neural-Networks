import kagglehub
import shutil
from pathlib import Path

# Define the target directory
TARGET_DIR = Path("/home/coder/IKT469-1-26V-Deep-Neural-Networks/assignment_1/data/winedataset")

path = kagglehub.dataset_download("yasserh/wine-quality-dataset")

print(f"Dataset downloaded to: {path}")

# Copy all files from the downloaded path to our target directory
source_path = Path(path)
for file in source_path.glob("*"):
    if file.is_file():
        target_file = TARGET_DIR / file.name
        shutil.copy2(file, target_file)
        print(f"Copied: {file.name}")

print(f"\nDataset saved to: {TARGET_DIR}")
print(f"Files in directory: {list(TARGET_DIR.glob('*'))}")
