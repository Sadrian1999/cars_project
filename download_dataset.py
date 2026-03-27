import shutil
from pathlib import Path

import kagglehub

path = Path(kagglehub.dataset_download("austinreese/craigslist-carstrucks-data"))

csv_file = path / "vehicles.csv"

dst = Path("./data/raw/vehicles.csv")
dst.parent.mkdir(parents=True, exist_ok=True)

shutil.move(csv_file, dst)
