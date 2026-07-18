"""
Streamlit demo app for CatBoost German Used-Car Price Predictor.

Loads the trained CatBoost model (saved via save_model), the sklearn-compatible
preprocessor pipeline, and the custom feature engineer — all serialised as
.pkl / .cbm in the models/ directory.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt

from catboost import CatBoostRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "catboost_model.cbm"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_ENGINEER_PATH = MODEL_DIR / "feature_engineer.pkl"


# ---------------------------------------------------------------------------
# Cached resource loading  (only loads once per session / change)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Deserialise the three artefacts needed for inference."""
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    preprocessor = joblib.load(str(PREPROCESSOR_PATH))
    fe = joblib.load(str(FEATURE_ENGINEER_PATH))

    return model, preprocessor, fe


# ---------------------------------------------------------------------------
# Constants derived from the training dataset
# ---------------------------------------------------------------------------

MAKES = [
    "Audi", "BMW", "Ford", "Mercedes-Benz", "Opel",
    "Porsche", "Seat", "Skoda", "Volkswagen",
]

MODELS_BY_MAKE: dict[str, list[str]] = {
    "BMW":               ["1er", "2er", "3er", "4er", "5er", "6er", "7er",
                          "X1", "X3", "X5", "X6", "Z4", "i4", "iX"],
    "Mercedes-Benz":     ["A-Klasse", "B-Klasse", "C-Klasse", "E-Klasse",
                          "S-Klasse", "CLA", "GLA", "GLC", "GLE", "GLS",
                          "CLS", "SLK", "EQA", "EQS"],
    "Audi":              ["A1", "A3", "A4", "A5", "A6", "A7", "A8",
                          "Q2", "Q3", "Q5", "Q7", "Q8", "e-tron"],
    "Volkswagen":        ["Polo", "Golf", "Golf GTI", "Passat", "Passat CC",
                          "Tiguan", "Tiguan R-Line", "Touran", "T-Roc",
                          "T-Cross", "ID.3", "ID.4", "Arteon"],
    "Opel":              ["Corsa", "Astra", "Insignia", "Mokka",
                          "Crossland", "Grandland", "Zafira"],
    "Porsche":           ["911", "Cayenne", "Macan", "Panamera",
                          "Taycan", "Cayman", "Boxster"],
    "Ford":              ["Fiesta", "Focus", "Focus ST", "Kuga",
                          "Puma", "Mustang", "Mondeo", "S-Max"],
    "Seat":              ["Ibiza", "Leon", "Leon Cupra",
                          "Arona", "Ateca", "Tarraco"],
    "Skoda":             ["Fabia", "Octavia", "Octavia RS", "Superb",
                          "Superb L&K", "Kodiaq", "Kodiaq RS",
                          "Karoq", "Kamiq", "Enyaq"],
}

FUEL_TYPES = ["diesel", "petrol", "hybrid", "electric"]
TRANSMISSIONS = ["manual", "automatic"]
DRIVE_TYPES = ["front", "rear", "all"]
COLORS = ["white", "black", "silver", "blue", "red", "grey", "green"]
CITIES = [
    "Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Stuttgart",
    "Duesseldorf", "Leipzig", "Dortmund", "Essen", "Bremen", "Dresden",
    "Hannover", "Nuernberg", "Bielefeld", "Bonn", "Muenster",
]
SELLER_TYPES = ["private", "dealer"]
CONDITIONS = ["excellent", "good", "fair"]
SERVICE_HISTORIES = ["full", "partial", "none"]
TRIMS = ["base", "sport", "m_sport", "amg", "s_line"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚗 German Used-Car Price Estimator")
st.markdown(
    """
    Enter the details of a vehicle below and get an **instant market-value
    estimate** powered by a CatBoost gradient-boosting model trained on
    synthetic German used-car data.
    """
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

with st.spinner("Loading model …"):
    model, preprocessor, feature_engineer = load_artifacts()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

st.subheader("Vehicle Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Make", MAKES)
        model_name = st.selectbox("Model", MODELS_BY_MAKE[make])
        year = st.slider("Year", 1996, 2025, 2018)
        mileage = st.number_input(
            "Mileage (km)", min_value=0, max_value=400_000,
            value=80_000, step=1000,
        )
        fuel = st.selectbox("Fuel Type", FUEL_TYPES)
        engine_size = st.number_input(
            "Engine Size (L)", min_value=0.0, max_value=6.0,
            value=2.0, step=0.1, format="%.1f",
        )
        horsepower = st.number_input(
            "Horsepower (PS)", min_value=50, max_value=700,
            value=150, step=5,
        )
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
        drive_type = st.selectbox("Drive Type", DRIVE_TYPES)

    with col2:
        color = st.selectbox("Color", COLORS)
        condition = st.selectbox("Condition", CONDITIONS)
        accident = st.selectbox("Accident Reported", ["no", "yes"])
        service_history = st.selectbox("Service History", SERVICE_HISTORIES)
        trim = st.selectbox("Trim Level", TRIMS)
        owners = st.slider("Previous Owners", 1, 10, 2)
        doors = st.slider("Doors", 2, 5, 5)
        tuv_months = st.slider("TÜV Months Remaining", 0, 36, 12)
        consumption = st.number_input(
            "Fuel Consumption (L/100km)",
            min_value=0.1, max_value=50.0, value=7.0, step=0.5,
        )
        city = st.selectbox("City", CITIES)
        seller_type = st.selectbox("Seller Type", SELLER_TYPES)

    submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if submitted:
    input_dict = {
        "make": make,
        "model": model_name,
        "year": year,
        "mileage": mileage,
        "engine_size": engine_size,
        "horsepower": horsepower,
        "fuel": fuel,
        "transmission": transmission,
        "drive_type": drive_type,
        "color": color,
        "city": city,
        "seller_type": seller_type,
        "doors": doors,
        "owners": owners,
        "accident": accident,
        "condition": condition,
        "service_history": service_history,
        "trim": trim,
        "tuv_months": tuv_months,
        "consumption_l_per_100km": consumption,
    }

    df = pd.DataFrame([input_dict])

    df = preprocessor.transform(df)
    df = feature_engineer.transform(df)

    pred_log = model.predict(df)[0]
    pred_price = float(np.expm1(pred_log))

    lower = pred_price * 0.85
    upper = pred_price * 1.15

    # ------------------------------------------------------------------
    # Result panel
    # ------------------------------------------------------------------
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.metric(
            label="Predicted Market Price",
            value=f"€{pred_price:,.0f}",
            delta=None,
        )

    with col_right:
        st.markdown(
            f"""
            **Confidence Range (≈ ±15%)**  
            €{lower:,.0f} – €{upper:,.0f}
            """
        )

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("What drives this prediction?")

    FEATURE_LABELS: dict[str, str] = {
        "vehicle_age": "Vehicle Age",
        "vehicle_age_squared": "Age²",
        "log_mileage": "Log Mileage",
        "km_per_year": "km / Year",
        "engine_power_ratio": "Power / Displacement",
        "engine_size_log": "Log Engine Size",
        "luxury_brand": "Luxury Brand",
        "premium_segment": "Premium Segment",
        "age_mileage_interaction": "Age × Mileage",
        "power_depreciation": "Power / Age",
        "condition_multiplier": "Condition",
        "accident_multiplier": "Accident History",
        "service_multiplier": "Service History",
        "trim_multiplier": "Trim Level",
        "tuv_multiplier": "TÜV Status",
        "consumption_normalized": "Fuel Efficiency",
        "market_adjustment": "Market Adjustment",
        "sqrt_mileage": "√Mileage",
        "inverse_age": "1 / Age",
        "make": "Make",
        "model": "Model",
        "year": "Year",
        "mileage": "Mileage",
        "engine_size": "Engine Size",
        "horsepower": "Horsepower",
        "fuel": "Fuel Type",
        "transmission": "Transmission",
        "drive_type": "Drive Type",
        "color": "Color",
        "city": "City",
        "seller_type": "Seller Type",
        "doors": "Doors",
        "owners": "Previous Owners",
        "accident": "Accident",
        "condition": "Condition (raw)",
        "service_history": "Service History (raw)",
        "trim": "Trim (raw)",
        "tuv_months": "TÜV Months",
        "consumption_l_per_100km": "Consumption",
    }

    feat_names = model.feature_names_
    feat_imp = model.get_feature_importance()

    imp_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": feat_imp})
        .assign(Label=lambda df: df["Feature"].map(FEATURE_LABELS))
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X("Importance:Q", title="Importance"),
            y=alt.Y("Label:N", title=None, sort="-x"),
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Feature importance shows how much each variable contributes to the "
        "model's decisions. The top features have the strongest influence on "
        "the predicted price."
    )

    # ------------------------------------------------------------------
    # Optional: full input summary
    # ------------------------------------------------------------------
    with st.expander("Show input summary"):
        st.json(input_dict)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    Built with [CatBoost](https://catboost.ai/) • [Streamlit](https://streamlit.io/) • [scikit-learn](https://scikit-learn.org/)
    """
)
