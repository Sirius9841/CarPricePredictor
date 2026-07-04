from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostRegressor

from src.utils import logger
from src.config import PathConfig, ModelConfig


MAKES = ["BMW", "Mercedes-Benz", "Audi", "Volkswagen", "Opel", "Porsche", "Ford", "Seat", "Skoda"]
FUEL_TYPES = ["diesel", "petrol", "hybrid", "electric"]
TRANSMISSIONS = ["manual", "automatic"]
DRIVE_TYPES = ["front", "rear", "all"]
CONDITIONS = ["excellent", "good", "fair"]
SERVICE_HISTORIES = ["full", "partial", "none"]
TRIMS = ["base", "sport", "m_sport", "amg", "s_line"]
COLORS = ["white", "black", "silver", "blue", "red", "grey", "green"]
SELLER_TYPES = ["private", "dealer"]
CITIES = ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Stuttgart",
          "Duesseldorf", "Leipzig", "Dortmund", "Essen", "Bremen", "Dresden",
          "Hannover", "Nuernberg", "Bielefeld", "Bonn", "Muenster"]


# =========================
# Prediction System
# =========================

class CarPricePredictor:
    """
    Production inference system for used car price prediction.
    """

    def __init__(self, path_config: PathConfig, model_config: ModelConfig):
        self.path_config = path_config
        self.model_config = model_config

        self.model: CatBoostRegressor = self._load_model()
        self.preprocessor = self._load_preprocessor()
        self.feature_engineer = self._load_feature_engineer()

    # =========================
    # Load artifacts
    # =========================

    def _load_model(self) -> CatBoostRegressor:
        model_path = os.path.join(self.path_config.model_dir, "catboost_model.cbm")

        model = CatBoostRegressor()
        model.load_model(model_path)

        logger.info("Model loaded successfully")
        return model

    def _load_preprocessor(self):
        path = os.path.join(self.path_config.model_dir, "preprocessor.pkl")
        logger.info("Preprocessor loaded")
        return joblib.load(path)

    def _load_feature_engineer(self):
        path = os.path.join(self.path_config.model_dir, "feature_engineer.pkl")
        logger.info("Feature engineer loaded")
        return joblib.load(path)

    # =========================
    # Core Prediction
    # =========================

    def predict(self, input_data: dict) -> dict:
        """
        Predict price for a single vehicle input.
        """

        df = pd.DataFrame([input_data])

        df = self.preprocessor.transform(df)

        df = self.feature_engineer.transform(df)

        pred_log = self.model.predict(df)[0]

        if self.model_config.log_target:
            pred = float(np.expm1(pred_log))
        else:
            pred = float(pred_log)

        lower = pred * 0.85
        upper = pred * 1.15

        return {
            "predicted_price": round(pred, 2),
            "confidence_interval": (round(lower, 2), round(upper, 2)),
        }


# =========================
# CLI helpers
# =========================

def _prompt(label: str, options: list[str] | None = None, default: str | None = None) -> str:
    hint = ""
    if options:
        hint = f" [{', '.join(options)}]"
    elif default is not None:
        hint = f" (e.g. {default})"
    prompt_text = f"{label}{hint}: "
    val = input(prompt_text).strip()
    if not val and default is not None:
        return default
    return val


def _prompt_int(label: str, default: int | None = None, min_val: int | None = None, max_val: int | None = None) -> int:
    while True:
        raw = _prompt(label, default=str(default) if default else None)
        try:
            v = int(raw)
            if min_val is not None and v < min_val:
                print(f"  Must be >= {min_val}, got {v}")
                continue
            if max_val is not None and v > max_val:
                print(f"  Must be <= {max_val}, got {v}")
                continue
            return v
        except ValueError:
            print(f"  Invalid integer: '{raw}'")


def _prompt_float(label: str, default: float | None = None) -> float:
    while True:
        raw = _prompt(label, default=str(default) if default else None)
        try:
            return float(raw)
        except ValueError:
            print(f"  Invalid number: '{raw}'")


def _prompt_choice(label: str, options: list[str], default_idx: int = 0) -> str:
    while True:
        raw = _prompt(label, options=options).lower()
        if not raw:
            return options[default_idx]
        matches = [o for o in options if o.lower() == raw or o.lower().startswith(raw)]
        if len(matches) == 1:
            return matches[0]
        print(f"  Options: {', '.join(options)}")


# =========================
# CLI Interface
# =========================

def run_cli():
    """
    Interactive user interface for real predictions.
    """

    path_config = PathConfig()
    model_config = ModelConfig()

    predictor = CarPricePredictor(path_config, model_config)

    print("\n" + "=" * 55)
    print("  German Used-Car Price Estimator")
    print("  Press Enter to accept defaults shown in brackets")
    print("=" * 55)

    d = {}

    d["make"] = _prompt_choice("Make", MAKES)
    d["model"] = _prompt("Model", default="3er" if d["make"] == "BMW" else "Golf")

    d["year"] = _prompt_int("Year", default=2018, min_val=1996, max_val=2026)
    current_year = 2026
    age = current_year - d["year"]
    suggested_mileage = int(age * 15000)
    d["mileage"] = _prompt_float("Mileage (km)", default=float(suggested_mileage))

    d["fuel"] = _prompt_choice("Fuel type", FUEL_TYPES, default_idx=0)
    d["engine_size"] = _prompt_float("Engine size (L)", default=2.0)
    d["horsepower"] = _prompt_int("Horsepower (PS)", default=150, min_val=50, max_val=700)

    d["transmission"] = _prompt_choice("Transmission", TRANSMISSIONS, default_idx=1)
    d["drive_type"] = _prompt_choice("Drive type", DRIVE_TYPES, default_idx=0)
    d["color"] = _prompt_choice("Color", COLORS, default_idx=0)

    d["condition"] = _prompt_choice("Condition", CONDITIONS, default_idx=1)
    d["accident"] = _prompt_choice("Accident reported", ["no", "yes"], default_idx=0)
    d["service_history"] = _prompt_choice("Service history", SERVICE_HISTORIES, default_idx=0)
    d["trim"] = _prompt_choice("Trim level", TRIMS, default_idx=0)
    d["tuv_months"] = _prompt_int("TUV months remaining", default=12, min_val=0, max_val=36)
    d["consumption_l_per_100km"] = _prompt_float("Fuel consumption (L/100km)", default=7.0)

    d["owners"] = _prompt_int("Previous owners", default=2, min_val=1, max_val=10)
    d["doors"] = _prompt_int("Doors", default=5, min_val=2, max_val=5)
    d["seller_type"] = _prompt_choice("Seller type", SELLER_TYPES, default_idx=1)
    d["city"] = _prompt("City", default="Berlin")

    result = predictor.predict(d)

    print("\n" + "=" * 55)
    print("  PRICE ESTIMATE")
    print("=" * 55)
    print(f"  Predicted Price:      EUR {result['predicted_price']:>10,.2f}")
    print(f"  Confidence Range:     EUR {result['confidence_interval'][0]:>10,.2f} - EUR {result['confidence_interval'][1]:>,.2f}")
    print("=" * 55)


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    run_cli()