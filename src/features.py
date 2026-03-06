import re

import numpy as np
import pandas as pd


def build_model_pattern(validator: pd.DataFrame) -> str:
    v = validator.copy()

    v["manufacturer"] = v["make"].astype(str).str.lower().str.strip()
    v["model"] = v["model"].astype(str).str.lower().str.strip()
    v = v.dropna(subset=["manufacturer", "model"])

    models = sorted(
        v["model"].dropna().unique(),
        key=len,
        reverse=True,
    )

    pattern = r"\b(" + "|".join(map(re.escape, models)) + r")\b"
    return pattern


def add_base_model(cars: pd.DataFrame, pattern: str) -> pd.DataFrame:
    cars = cars.copy()

    cars["manufacturer"] = cars["manufacturer"].astype(str).str.lower().str.strip()
    cars["model"] = cars["model"].astype(str).str.lower().str.strip()

    cars["base_model"] = cars["model"].str.extract(pattern, expand=False)
    return cars


def add_features(cars: pd.DataFrame) -> pd.DataFrame:
    cars = cars.copy()

    current_year = int(pd.to_numeric(cars["year"], errors="coerce").max())
    cars["age"] = current_year - cars["year"]
    cars = cars[(cars["age"] >= 0) & (cars["age"] <= 60)]

    cars["log_odometer"] = np.log1p(cars["odometer"])

    cars["odometer_per_year"] = cars["odometer"] / (cars["age"] + 1)
    cars.loc[~np.isfinite(cars["odometer_per_year"]), "odometer_per_year"] = np.nan
    cars["log_opy"] = np.log1p(cars["odometer_per_year"])

    cars["log_price"] = np.log1p(cars["price"])

    return cars
