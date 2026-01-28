import os
import shutil
from pathlib import Path

import kagglehub

src = Path(kagglehub.dataset_download("zaraks/pascal-voc-2007"))

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

dst = data_dir / "pascal-voc-2007"

if dst.exists():
    print("Dataset already available at:", dst)
else:
    try:
        os.symlink(src, dst, target_is_directory=True)
        print("Symlinked dataset to:", dst)
        print("-> points to:", src)
    except OSError:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print("Copied dataset to:", dst)

print("Path to dataset files:", dst)
