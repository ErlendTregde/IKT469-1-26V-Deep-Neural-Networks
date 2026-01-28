import os
import shutil
from pathlib import Path

import kagglehub

# Download dataset (cached by kagglehub)
src = Path(kagglehub.dataset_download("zaraks/pascal-voc-2007"))

# Put it under ./data
data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

dst = data_dir / "pascal-voc-2007"

# If it already exists, do nothing
if dst.exists():
    print("Dataset already available at:", dst)
else:
    # Prefer a symlink (fast, no extra disk). Fall back to copy on failure.
    try:
        os.symlink(src, dst, target_is_directory=True)
        print("Symlinked dataset to:", dst)
        print("-> points to:", src)
    except OSError:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print("Copied dataset to:", dst)

print("Path to dataset files:", dst)
