import shutil

import kagglehub

# Download latest version
path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")

shutil.move(path, "./data/raw/")
