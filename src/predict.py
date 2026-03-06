"""src/predict.py"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def load_models():
    mid = joblib.load(MODELS_DIR / "hgb_mid.joblib")
    low = joblib.load(MODELS_DIR / "hgb_low.joblib")
    high = joblib.load(MODELS_DIR / "hgb_high.joblib")
    return low, mid, high


def make_sample(
    age: float,
    odometer: float,
    manufacturer: str,
    base_model: str,
    condition: str,
    fuel: str,
    transmission: str,
    drive: str,
    type_: str,
    title_status: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": age,
                "log_odometer": np.log1p(odometer),
                "manufacturer": str(manufacturer).strip().lower(),
                "base_model": str(base_model).strip().lower(),
                "condition": str(condition).strip().lower(),
                "fuel": str(fuel).strip().lower(),
                "transmission": str(transmission).strip().lower(),
                "drive": str(drive).strip().lower(),
                "type": str(type_).strip().lower(),
                "title_status": str(title_status).strip().lower(),
            }
        ]
    )


def predict_range(sample_df: pd.DataFrame, low_model, mid_model, high_model):
    low = float(np.expm1(low_model.predict(sample_df)[0]))
    mid = float(np.expm1(mid_model.predict(sample_df)[0]))
    high = float(np.expm1(high_model.predict(sample_df)[0]))
    return low, mid, high
