from pathlib import Path
import shutil

import kagglehub

# Download the dataset and copy it into the data folder beside this script.
download_path = Path(kagglehub.dataset_download("ellipticco/elliptic-data-set"))
data_path = Path(__file__).resolve().parent / "data"
shutil.copytree(download_path, data_path, dirs_exist_ok=True)

print("Path to dataset files:", data_path)