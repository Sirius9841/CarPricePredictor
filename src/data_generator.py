from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import logger


SEED = 42

MODELS_BY_MAKE: dict[str, dict[str, float]] = {
    "BMW": {"1er": 0.70, "2er": 0.85, "3er": 1.00, "4er": 1.10, "5er": 1.35, "6er": 1.60, "7er": 2.10,
            "X1": 0.90, "X3": 1.20, "X5": 1.70, "X6": 1.80, "Z4": 1.40, "i4": 1.15, "iX": 2.00},
    "Mercedes-Benz": {"A-Klasse": 0.70, "B-Klasse": 0.75, "C-Klasse": 1.00, "E-Klasse": 1.40,
                      "S-Klasse": 2.30, "CLA": 0.90, "GLA": 0.85, "GLC": 1.15, "GLE": 1.65,
                      "GLS": 2.10, "CLS": 1.55, "SLK": 1.40, "EQA": 1.10, "EQS": 2.40},
    "Audi": {"A1": 0.60, "A3": 0.85, "A4": 1.00, "A5": 1.10, "A6": 1.45, "A7": 1.75, "A8": 2.30,
             "Q2": 0.80, "Q3": 0.95, "Q5": 1.20, "Q7": 1.65, "Q8": 1.85, "e-tron": 1.50},
    "Volkswagen": {"Polo": 0.50, "Golf": 0.75, "Golf GTI": 1.05, "Passat": 0.95, "Passat CC": 1.15,
                   "Tiguan": 1.05, "Tiguan R-Line": 1.20, "Touran": 0.90,
                   "T-Roc": 0.85, "T-Cross": 0.70, "ID.3": 0.90, "ID.4": 1.10, "Arteon": 1.25},
    "Opel": {"Corsa": 0.50, "Astra": 0.75, "Insignia": 0.95, "Mokka": 0.65, "Crossland": 0.70,
             "Grandland": 0.90, "Zafira": 0.80},
    "Porsche": {"911": 2.60, "Cayenne": 1.85, "Macan": 1.30, "Panamera": 2.10, "Taycan": 2.30,
                "Cayman": 1.60, "Boxster": 1.50},
    "Ford": {"Fiesta": 0.50, "Focus": 0.75, "Focus ST": 1.05, "Kuga": 0.85, "Puma": 0.70,
             "Mustang": 1.50, "Mondeo": 0.95, "S-Max": 0.90},
    "Seat": {"Ibiza": 0.50, "Leon": 0.80, "Leon Cupra": 1.10, "Arona": 0.65, "Ateca": 0.85,
             "Tarraco": 1.00},
    "Skoda": {"Fabia": 0.50, "Octavia": 0.80, "Octavia RS": 1.10, "Superb": 1.05, "Superb L&K": 1.20,
              "Kodiaq": 1.00, "Kodiaq RS": 1.20, "Karoq": 0.85, "Kamiq": 0.70, "Enyaq": 1.10},
}

MAKE_BASE_PRICES: dict[str, float] = {
    "BMW": 42000, "Mercedes-Benz": 44000, "Audi": 39000,
    "Volkswagen": 32000, "Opel": 25000, "Porsche": 75000,
    "Ford": 27000, "Seat": 24000, "Skoda": 26000,
}

BASE_MAKE_TRIMS: dict[str, list[str]] = {
    "BMW": ["base", "sport", "m_sport", "s_line"],
    "Mercedes-Benz": ["base", "sport", "amg", "s_line"],
    "Audi": ["base", "sport", "s_line"],
    "Volkswagen": ["base", "sport", "s_line"],
    "Opel": ["base", "sport"],
    "Porsche": ["base", "sport", "s_line"],
    "Ford": ["base", "sport"],
    "Seat": ["base", "sport"],
    "Skoda": ["base", "sport"],
}

GERMAN_CITIES: list[str] = [
    "Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt", "Stuttgart",
    "Duesseldorf", "Leipzig", "Dortmund", "Essen", "Bremen", "Dresden",
    "Hannover", "Nuernberg", "Bielefeld", "Bonn", "Muenster",
]

MAKE_CITY_PREFERENCES: dict[str, float] = {
    "BMW": 0.30, "Mercedes-Benz": 0.25, "Audi": 0.25,
    "Porsche": 0.40, "Tesla": 0.30,
}

SELLER_TYPES: list[str] = ["private", "dealer"]

FUEL_TYPES: list[str] = ["diesel", "petrol", "hybrid", "electric"]

TRANSMISSION_TYPES: list[str] = ["manual", "automatic"]

DRIVE_TYPES: list[str] = ["front", "rear", "all"]

COLORS: list[str] = ["white", "black", "silver", "blue", "red", "grey", "green"]

COLOR_FACTORS: dict[str, float] = {
    "white": 1.00, "black": 1.02, "silver": 1.01, "blue": 0.98,
    "red": 0.95, "grey": 0.99, "green": 0.92,
}

CONDITIONS: list[str] = ["excellent", "good", "fair"]

CONDITION_MULTIPLIERS: dict[str, float] = {
    "excellent": 1.15, "good": 1.0, "fair": 0.80,
}

SERVICE_HISTORIES: list[str] = ["full", "partial", "none"]

SERVICE_MULTIPLIERS: dict[str, float] = {
    "full": 1.08, "partial": 1.00, "none": 0.90,
}

TRIM_MULTIPLIERS: dict[str, float] = {
    "base": 1.00, "sport": 1.05, "m_sport": 1.12, "amg": 1.15, "s_line": 1.10,
}

ENGINE_CONFIGS: dict[str, list[tuple[float, float, float, float, str, float]]] = {
    "diesel": [(1.6, 90, 116, 4.5, "diesel", 1.05),
               (2.0, 110, 190, 5.0, "diesel", 1.05),
               (2.0, 140, 200, 5.5, "diesel", 1.05),
               (3.0, 180, 286, 6.5, "diesel", 1.05),
               (3.0, 210, 340, 7.0, "diesel", 1.05)],
    "petrol": [(1.0, 60, 75, 5.0, "petrol", 1.00),
               (1.2, 70, 90, 5.5, "petrol", 1.00),
               (1.4, 90, 125, 6.0, "petrol", 1.00),
               (1.5, 100, 150, 6.2, "petrol", 1.00),
               (1.6, 110, 160, 6.5, "petrol", 1.00),
               (2.0, 140, 210, 7.5, "petrol", 1.00),
               (2.0, 180, 280, 8.0, "petrol", 1.00),
               (3.0, 220, 370, 9.5, "petrol", 1.00),
               (4.0, 300, 460, 12.0, "petrol", 1.00)],
    "hybrid": [(1.5, 100, 160, 4.0, "hybrid", 1.15),
               (2.0, 140, 255, 4.5, "hybrid", 1.15),
               (2.5, 180, 305, 5.0, "hybrid", 1.15),
               (3.0, 210, 350, 5.5, "hybrid", 1.15)],
    "electric": [(0.0, 100, 170, 14.0, "electric", 1.30),
                 (0.0, 150, 250, 16.0, "electric", 1.30),
                 (0.0, 200, 350, 18.0, "electric", 1.30),
                 (0.0, 250, 476, 20.0, "electric", 1.30)],
}


def _set_seed(rng: np.random.Generator) -> None:
    np.random.seed(SEED)


def _validate_constraints(row: dict) -> dict:
    mileage = row["mileage"]
    condition = row["condition"]
    accident = row["accident"]
    age = row["vehicle_age"]
    fuel = row["fuel"]
    consumption = row.get("consumption_l_per_100km", 0)

    if condition == "excellent" and mileage > 180000 and accident == "no":
        row["condition"] = "good"
        row["mileage"] = int(mileage * 0.85)

    if condition == "excellent" and mileage > 250000:
        row["condition"] = "good"

    if condition == "fair" and mileage < 30000 and accident == "no":
        row["accident"] = "yes"

    if age < 2 and condition == "fair":
        row["condition"] = "good"

    if age > 15 and condition == "excellent":
        row["condition"] = "good"

    if fuel in ("diesel", "petrol", "hybrid") and consumption > 25:
        row["consumption_l_per_100km"] = round(consumption * 0.75, 1)

    if fuel == "electric" and consumption < 8:
        row["consumption_l_per_100km"] = round(consumption * 1.5, 1)

    return row


def _compute_price(row: dict) -> float:
    base_price = row["_base_make_price"] * row["_model_factor"]
    age = row["vehicle_age"]
    mileage = row["mileage"]
    condition = row["condition"]
    accident = row["accident"]
    service_history = row["service_history"]
    trim = row["trim"]
    tuv_months = row["tuv_months"]
    owners = row["owners"]
    fuel_factor = row["_fuel_factor"]
    transmission_factor = row["_transmission_factor"]
    drive_type_factor = row["_drive_type_factor"]
    color_factor = row["_color_factor"]

    depreciation = 0.35 * np.exp(-0.55 * age) + 0.40 * np.exp(-0.10 * age) + 0.25
    mileage_factor = 1.0 - 0.45 * (1.0 - np.exp(-mileage / 250000))

    is_luxury = row.get("_is_luxury", False)

    condition_mult = CONDITION_MULTIPLIERS[condition]
    accident_mult = 0.88 if accident == "yes" else 1.0
    service_mult = SERVICE_MULTIPLIERS[service_history]
    trim_mult = TRIM_MULTIPLIERS.get(trim, 1.0)
    tuv_mult = 0.85 if tuv_months < 3 else (1.0 if tuv_months <= 12 else 1.05)
    owners_penalty = np.exp(-0.06 * (owners - 1))

    # A) Luxury brands depreciate much harder — old prestige cars are worth little
    if is_luxury:
        luxury_depr_factor = 0.45 + 0.55 * np.exp(-0.15 * age)
        depreciation *= luxury_depr_factor

    # B) Interactions: luxury cars penalised extra for bad state
    luxury_interaction = 1.0
    if is_luxury:
        if condition == "fair":
            luxury_interaction *= 0.85
        if accident == "yes":
            luxury_interaction *= 0.88
        if age > 8 and mileage > 120000:
            luxury_interaction *= 0.88

    price = (
        base_price
        * depreciation
        * mileage_factor
        * condition_mult
        * accident_mult
        * service_mult
        * trim_mult
        * tuv_mult
        * fuel_factor
        * transmission_factor
        * drive_type_factor
        * color_factor
        * owners_penalty
        * luxury_interaction
    )

    market_noise = np.random.normal(1.0, 0.08)
    market_noise = np.clip(market_noise, 0.85, 1.15)
    price *= market_noise

    return max(round(price, 2), 500.0)


def _validate_price_floor(row: dict, price: float) -> float:
    if price < 1000 and row["vehicle_age"] < 5:
        price = float(np.random.uniform(3000, 8000))
    return price


def generate_synthetic_data(n: int = 10000) -> pd.DataFrame:
    np.random.seed(SEED)

    makes = list(MODELS_BY_MAKE.keys())
    luxury_brands = {"bmw", "mercedes-benz", "audi", "porsche"}
    rows: list[dict] = []

    for _ in range(n):
        make = str(np.random.choice(makes))
        model_options = MODELS_BY_MAKE[make]
        model_name = str(np.random.choice(list(model_options.keys())))
        model_factor = model_options[model_name]

        age = float(np.random.exponential(scale=6.0) + 1.0)
        age = min(age, 30.0)

        km_per_year = float(np.random.lognormal(mean=9.55, sigma=0.35))
        mileage = float(np.random.lognormal(
            mean=np.log(km_per_year * age + 1000), sigma=0.25
        ))
        mileage = min(mileage, 350000.0)

        min_mileage = max(100.0, age * 500.0)
        mileage = max(mileage, min_mileage)

        make_is_luxury = make.lower() in luxury_brands
        fuel_petrol_weight = 0.45
        fuel_diesel_weight = 0.35
        fuel_hybrid_weight = 0.12
        fuel_electric_weight = 0.08

        if mileage > 150000:
            fuel_diesel_weight = 0.55
            fuel_petrol_weight = 0.28
            fuel_hybrid_weight = 0.10
            fuel_electric_weight = 0.07
        elif age < 4:
            fuel_hybrid_weight = 0.20
            fuel_electric_weight = 0.15
            fuel_petrol_weight = 0.40
            fuel_diesel_weight = 0.25

        fuel = str(np.random.choice(
            FUEL_TYPES,
            p=[fuel_diesel_weight, fuel_petrol_weight, fuel_hybrid_weight, fuel_electric_weight]
        ))
        engine_configs = ENGINE_CONFIGS[fuel]
        if len(engine_configs) > 1:
            hp_values = [c[1] for c in engine_configs]
            hp_total = sum(hp_values)
            hp_weights = [h / hp_total for h in hp_values]
            engine_idx = int(np.random.choice(len(engine_configs), p=hp_weights))
        else:
            engine_idx = 0

        engine_size = engine_configs[engine_idx][0]
        hp = engine_configs[engine_idx][1]
        max_hp = engine_configs[engine_idx][2]
        consumption = engine_configs[engine_idx][3]
        fuel_factor = engine_configs[engine_idx][5]

        year = int(round(2026 - age))
        year = max(1996, min(year, 2026))

        transmission = str(np.random.choice(TRANSMISSION_TYPES, p=[0.40, 0.60]))
        if age > 12:
            transmission = str(np.random.choice(TRANSMISSION_TYPES, p=[0.60, 0.40]))
        elif age < 4:
            transmission = str(np.random.choice(TRANSMISSION_TYPES, p=[0.25, 0.75]))

        drive_type = "front"
        if make_is_luxury or make == "Porsche":
            drive_type = str(np.random.choice(DRIVE_TYPES, p=[0.20, 0.35, 0.45]))
        elif make in ("BMW",):
            drive_type = str(np.random.choice(DRIVE_TYPES, p=[0.10, 0.60, 0.30]))
        else:
            drive_type = str(np.random.choice(DRIVE_TYPES, p=[0.55, 0.10, 0.35]))

        drive_type_factors = {"front": 1.0, "rear": 1.03, "all": 1.06}
        drive_type_factor = drive_type_factors[drive_type]

        if age < 4 and make_is_luxury:
            transmission = str(np.random.choice(TRANSMISSION_TYPES, p=[0.15, 0.85]))

        color = str(np.random.choice(COLORS, p=[0.25, 0.20, 0.18, 0.12, 0.10, 0.12, 0.03]))
        color_factor = COLOR_FACTORS[color]

        accident_prob = float(np.clip(0.05 + age * 0.015, 0.0, 0.60))
        accident = "yes" if np.random.random() < accident_prob else "no"

        owners = min(int(np.random.geometric(0.35)), 6)

        condition_prob_base = np.clip(1.0 - mileage / 300000, 0.0, 1.0)
        if accident == "yes":
            condition_prob_base *= 0.6
        condition_noise = float(np.random.uniform(-0.15, 0.15))
        condition_score = np.clip(condition_prob_base + condition_noise, 0.0, 1.0)

        if condition_score > 0.65:
            condition = "excellent"
        elif condition_score > 0.30:
            condition = "good"
        else:
            condition = "fair"

        service_prob = float(np.clip(0.85 - age * 0.025, 0.10, 0.95))
        if condition == "excellent":
            service_prob = float(np.clip(service_prob + 0.15, 0.0, 0.98))
        elif condition == "fair":
            service_prob = float(np.clip(service_prob - 0.20, 0.0, 0.70))
        service_rnd = float(np.random.random())
        if service_rnd < service_prob:
            service_history = "full"
        elif service_rnd < service_prob + (1.0 - service_prob) * 0.6:
            service_history = "partial"
        else:
            service_history = "none"

        available_trims = BASE_MAKE_TRIMS.get(make, ["base", "sport"])
        if age < 4:
            trim_weights = [0.15, 0.30, 0.30, 0.15, 0.10][:len(available_trims)]
        elif age < 10:
            trim_weights = [0.30, 0.35, 0.20, 0.10, 0.05][:len(available_trims)]
        else:
            trim_weights = [0.50, 0.30, 0.15, 0.05, 0.00][:len(available_trims)]
        trim_weight_sum = float(sum(trim_weights))
        if trim_weight_sum > 0:
            trim_weights = [w / trim_weight_sum for w in trim_weights]
        else:
            trim_weights = [1.0 / len(available_trims)] * len(available_trims)
        trim = str(np.random.choice(available_trims, p=trim_weights))

        tuv_months = float(np.clip(
            np.random.uniform(0, 24) * (1.0 - min(age / 20.0, 0.5))
            + np.random.normal(0, 3),
            0, 36
        ))

        consumption_val = float(max(consumption + np.random.normal(0, 0.6), 1.0))

        seller_type = str(np.random.choice(SELLER_TYPES, p=[0.35, 0.65]))

        luxury_city_prob = MAKE_CITY_PREFERENCES.get(make, 0.0)
        if np.random.random() < luxury_city_prob:
            city = str(np.random.choice(
                ["Munich", "Stuttgart", "Duesseldorf", "Frankfurt", "Hamburg", "Berlin"]
            ))
        else:
            city = str(np.random.choice(GERMAN_CITIES))

        doors = int(np.random.choice([3, 4, 5], p=[0.05, 0.60, 0.35]))

        transmission_factor = 0.95 if transmission == "manual" else 1.02

        row = {
            "make": make,
            "model": model_name,
            "_model_factor": model_factor,
            "year": year,
            "vehicle_age": age,
            "mileage": int(round(mileage)),
            "engine_size": engine_size,
            "horsepower": int(round(hp)),
            "max_horsepower": int(round(max_hp)),
            "fuel": fuel,
            "_fuel_factor": fuel_factor,
            "transmission": transmission,
            "_transmission_factor": transmission_factor,
            "drive_type": drive_type,
            "_drive_type_factor": drive_type_factor,
            "color": color,
            "_color_factor": color_factor,
            "city": city,
            "seller_type": seller_type,
            "doors": doors,
            "owners": owners,
            "accident": accident,
            "condition": condition,
            "service_history": service_history,
            "trim": trim,
            "tuv_months": round(tuv_months, 1),
            "consumption_l_per_100km": round(consumption_val, 1),
            "_base_make_price": MAKE_BASE_PRICES.get(make, 25000),
            "_is_luxury": make_is_luxury,
        }

        row = _validate_constraints(row)

        price = _compute_price(row)
        row["price"] = _validate_price_floor(row, price)

        clean_row = {
            "price": row["price"],
            "make": row["make"],
            "model": row["model"],
            "year": row["year"],
            "mileage": row["mileage"],
            "engine_size": row["engine_size"],
            "horsepower": row["horsepower"],
            "fuel": row["fuel"],
            "transmission": row["transmission"],
            "drive_type": row["drive_type"],
            "color": row["color"],
            "city": row["city"],
            "seller_type": row["seller_type"],
            "doors": row["doors"],
            "owners": row["owners"],
            "accident": row["accident"],
            "condition": row["condition"],
            "service_history": row["service_history"],
            "trim": row["trim"],
            "tuv_months": row["tuv_months"],
            "consumption_l_per_100km": row["consumption_l_per_100km"],
        }
        rows.append(clean_row)

    df = pd.DataFrame(rows)
    logger.info(
        f"Generated {n} synthetic German used-car listings "
        f"({df['make'].nunique()} makes, {df['model'].nunique()} models)"
    )
    return df


def generate_and_save_csv(n: int = 10000, path: str = "data/car_data.csv") -> None:
    import os
    df = generate_synthetic_data(n=n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {n} records to {path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="German used-car market simulator")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/car_data.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_and_save_csv(n=args.samples, path=args.output)
