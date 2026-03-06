import pandas as pd


def clean_raw(cars: pd.DataFrame) -> pd.DataFrame:
    cars = cars.copy()

    cars = cars.dropna(subset=["manufacturer", "model", "price", "year", "odometer"])
    cars = cars[cars["price"] > 0]
    cars = cars[(cars["price"] >= 200) & (cars["price"] <= 1_000_000)]
    cars = cars[(cars["odometer"] >= 0) & (cars["odometer"] <= 600_000)]
    cars = cars[(cars["year"] >= 1900) & (cars["year"] <= 2020)]

    cars["title_status"] = cars["title_status"].fillna("unknown")

    cars["manufacturer"] = cars["manufacturer"].astype(str).str.lower().str.strip()
    cars["model"] = cars["model"].astype(str).str.lower().str.strip()

    return cars
