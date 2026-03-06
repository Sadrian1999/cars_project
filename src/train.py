import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.clean import clean_raw
from src.features import add_base_model, add_features, build_model_pattern

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "vehicles.csv"
VALIDATOR_CSV = ROOT / "data" / "external" / "validator_set.csv"
MODELS_DIR = ROOT / "models"

CATEGORICAL = [
    "manufacturer",
    "base_model",
    "condition",
    "fuel",
    "transmission",
    "drive",
    "type",
    "title_status",
]
NUMERIC = ["age", "log_odometer"]


def main() -> None:
    start_time = time.time()
    print("🚀 Training started...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Loading data...")
    cars = pd.read_csv(DATA_RAW)
    validator = pd.read_csv(VALIDATOR_CSV)

    print("🧹 Cleaning data...")
    cars = clean_raw(cars)
    print("   rows after cleaning:", len(cars))

    print("🧠 Building base_model...")
    pattern = build_model_pattern(validator)
    cars = add_base_model(cars, pattern)
    cars = cars.dropna(subset=["base_model"])
    print("   rows after base_model:", len(cars))

    print("⚙️ Feature engineering...")
    cars = add_features(cars)
    print("   rows after features:", len(cars))

    print("📊 Train-test split...")
    X = cars[CATEGORICAL + NUMERIC]
    y = cars["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )

    hgb_mid = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )

    hgb_low = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.1,
                    max_iter=300,
                    learning_rate=0.05,
                    random_state=42,
                ),
            ),
        ]
    )

    hgb_high = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.9,
                    max_iter=300,
                    learning_rate=0.05,
                    random_state=42,
                ),
            ),
        ]
    )

    print("🌳 Training mid model...")
    hgb_mid.fit(X_train, y_train)

    print("📉 Training lower quantile...")
    hgb_low.fit(X_train, y_train)

    print("📈 Training upper quantile...")
    hgb_high.fit(X_train, y_train)

    print("📏 Evaluating...")
    y_pred = hgb_mid.predict(X_test)

    rmse_log = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_log = float(mean_absolute_error(y_test, y_pred))

    y_test_price = np.expm1(y_test)
    y_pred_price = np.expm1(y_pred)
    mae_dollar = float(mean_absolute_error(y_test_price, y_pred_price))

    print("💾 Saving models...")
    joblib.dump(hgb_mid, MODELS_DIR / "hgb_mid.joblib")
    joblib.dump(hgb_low, MODELS_DIR / "hgb_low.joblib")
    joblib.dump(hgb_high, MODELS_DIR / "hgb_high.joblib")

    metrics = {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "mae_dollar": mae_dollar,
        "n_rows": int(len(cars)),
        "features": {"categorical": CATEGORICAL, "numeric": NUMERIC},
    }
    (MODELS_DIR / "metrics.json").write_text(pd.Series(metrics).to_json(), encoding="utf-8")

    print("✅ Done.")
    print("HGB RMSE (log):", rmse_log)
    print("HGB MAE (log):", mae_log)
    print("HGB MAE ($):", round(mae_dollar, 2))

    end_time = time.time()
    print(f"⏱ Total training time: {round((end_time - start_time)/60, 2)} minutes")


if __name__ == "__main__":
    main()
