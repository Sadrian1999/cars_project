import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

MODELS_DIR = ROOT / "models"
VALIDATOR_CSV = ROOT / "data" / "external" / "validator_set.csv"
REF_YEAR = 2022

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


@st.cache_resource
def load_models():
    hgb_mid = joblib.load(MODELS_DIR / "hgb_mid.joblib")
    hgb_low = joblib.load(MODELS_DIR / "hgb_low.joblib")
    hgb_high = joblib.load(MODELS_DIR / "hgb_high.joblib")
    return hgb_mid, hgb_low, hgb_high


@st.cache_data
def load_validator():
    v = pd.read_csv(VALIDATOR_CSV)
    v = v.dropna(subset=["make", "model"])
    v["manufacturer"] = v["make"].astype(str).str.lower().str.strip()
    v["model"] = v["model"].astype(str).str.lower().str.strip()
    v = v.dropna(subset=["manufacturer", "model"])
    return v[["manufacturer", "model"]]


@st.cache_data
def options_from_validator(v: pd.DataFrame):
    makes = sorted(v["manufacturer"].unique().tolist())
    by_make = (
        v.groupby("manufacturer")["model"]
        .apply(lambda s: sorted(set(s.tolist()), key=len))
        .to_dict()
    )
    return makes, by_make


def make_input_df(
    manufacturer: str,
    base_model: str,
    year: int,
    odometer: int,
    condition: str,
    fuel: str,
    transmission: str,
    drive: str,
    car_type: str,
    title_status: str,
) -> pd.DataFrame:
    age = REF_YEAR - int(year)
    log_odometer = float(np.log1p(max(int(odometer), 0)))

    row = {
        "manufacturer": str(manufacturer).lower().strip(),
        "base_model": str(base_model).lower().strip(),
        "condition": condition if condition != "" else np.nan,
        "fuel": fuel if fuel != "" else np.nan,
        "transmission": transmission if transmission != "" else np.nan,
        "drive": drive if drive != "" else np.nan,
        "type": car_type if car_type != "" else np.nan,
        "title_status": title_status if title_status != "" else np.nan,
        "age": age,
        "log_odometer": log_odometer,
    }
    return pd.DataFrame([row])[CATEGORICAL + NUMERIC]


def predict_range(hgb_mid, hgb_low, hgb_high, X: pd.DataFrame):
    pred_mid = float(hgb_mid.predict(X)[0])
    pred_low = float(hgb_low.predict(X)[0])
    pred_high = float(hgb_high.predict(X)[0])

    mid_price = float(np.expm1(pred_mid))
    low_price = float(np.expm1(pred_low))
    high_price = float(np.expm1(pred_high))

    low_price, high_price = (min(low_price, high_price), max(low_price, high_price))
    return low_price, mid_price, high_price


st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor")

hgb_mid, hgb_low, hgb_high = load_models()
validator = load_validator()
makes, models_by_make = options_from_validator(validator)

col1, col2 = st.columns(2)
with col1:
    manufacturer = st.selectbox("Gyártó (manufacturer)", makes, index=0)
with col2:
    base_models = models_by_make.get(manufacturer, [])
    base_model = st.selectbox("Modell (base_model)", base_models, index=0 if base_models else None)

year = st.number_input("Gyártási év", min_value=1900, max_value=REF_YEAR, value=2014, step=1)
odometer = st.number_input(
    "Kilométeróra (odometer)", min_value=0, max_value=2_000_000, value=120_000, step=1_000
)

condition = st.selectbox(
    "Állapot (condition)",
    ["", "new", "like new", "excellent", "good", "fair", "salvage"],
    index=0,
)
fuel = st.selectbox(
    "Üzemanyag (fuel)", ["", "gas", "diesel", "hybrid", "electric", "other"], index=0
)
transmission = st.selectbox("Váltó (transmission)", ["", "automatic", "manual", "other"], index=0)
drive = st.selectbox("Hajtás (drive)", ["", "fwd", "rwd", "4wd"], index=0)
car_type = st.selectbox(
    "Típus (type)",
    [
        "",
        "sedan",
        "SUV",
        "truck",
        "pickup",
        "coupe",
        "hatchback",
        "wagon",
        "van",
        "convertible",
        "other",
    ],
    index=0,
)
title_status = st.selectbox(
    "Okmány (title_status)",
    ["", "clean", "rebuilt", "lien", "salvage", "missing", "parts only"],
    index=0,
)

if st.button("Ár becslése"):
    if manufacturer is None or manufacturer == "" or base_model is None or base_model == "":
        st.error("Válassz gyártót és modellt.")
    else:
        X = make_input_df(
            manufacturer=manufacturer,
            base_model=base_model,
            year=int(year),
            odometer=int(odometer),
            condition=condition,
            fuel=fuel,
            transmission=transmission,
            drive=drive,
            car_type=car_type,
            title_status=title_status,
        )
        low_p, mid_p, high_p = predict_range(hgb_mid, hgb_low, hgb_high, X)
        st.subheader("Eredmény")
        st.markdown(f"## 💰 ${mid_p:,.0f}")
        st.write(f"Intervallum (10–90%): ${low_p:,.0f} – ${high_p:,.0f}")
