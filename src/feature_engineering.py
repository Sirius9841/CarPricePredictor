from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from src.utils import logger


# =========================
# Main Feature Pipeline
# =========================

class FeatureEngineer:
    """
    Production-grade feature engineering pipeline for used car price prediction.
    """

    def __init__(
            self,
            luxury_brands: Optional[List[str]] = None,
    ):
        self.luxury_brands = set(brand.lower() for brand in (luxury_brands or []))
        self.fitted = False

        # State tracking to prevent inference data leakage and map categories safely
        self.categorical_mappings: dict[str, dict[str, int]] = {}

    # =========================
    # Public API
    # =========================

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Training-time feature engineering. Learns statistical bounds and mappings.
        """
        df = df.copy()

        # Run core feature logic
        df = self._create_vehicle_age(df)
        df = self._create_usage_features(df)
        df = self._create_engine_features(df)
        df = self._create_brand_features(df)
        df = self._create_market_features(df)
        df = self._create_interaction_features(df)
        df = self._create_german_market_features(df)
        df = self._apply_nonlinear_transforms(df)

        # Learn Categorical Mappings (NEW: Ensures any tree model can read your data)
        # We find all remaining string/object columns to encode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            # Create a dictionary mapping each unique string to a distinct integer index
            unique_vals = df[col].dropna().unique()
            self.categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

        # Apply the learned mappings to the training set
        df = self._encode_categoricals(df)

        self.fitted = True
        logger.info("Feature engineering fit_transform completed successfully.")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inference-time feature engineering (deterministic, using saved state).
        """
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform.")

        df = df.copy()

        # Run core feature logic (identical to fit)
        df = self._create_vehicle_age(df)
        df = self._create_usage_features(df)
        df = self._create_engine_features(df)
        df = self._create_brand_features(df)
        df = self._create_market_features(df)
        df = self._create_interaction_features(df)
        df = self._create_german_market_features(df)
        df = self._apply_nonlinear_transforms(df)

        # Apply the exact same mappings saved during training
        df = self._encode_categoricals(df)

        logger.info("Feature engineering transform completed successfully.")
        return df

    # =========================
    # Helper: Safe Encoding
    # =========================

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes string text to integers safely using training-time vocabulary."""
        for col, mapping in self.categorical_mappings.items():
            if col in df.columns:
                # Convert to string to avoid mixed-type issues, map values,
                # and fill completely new/unseen values with -1 (Unseen Category Flag)
                df[col] = df[col].astype(str).str.lower().map(mapping).fillna(-1).astype(int)
        return df

    # =========================
    # 1. Vehicle Age Features
    # =========================

    def _create_vehicle_age(self, df: pd.DataFrame) -> pd.DataFrame:
        if "year" not in df.columns:
            return df

        current_year = 2026
        df["vehicle_age"] = current_year - df["year"]

        df["vehicle_age"] = df["vehicle_age"].clip(lower=0, upper=80)
        df["vehicle_age_squared"] = df["vehicle_age"] ** 2

        return df

    # =========================
    # 2. Usage Features
    # =========================

    def _create_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "mileage" not in df.columns:
            return df

        df["log_mileage"] = np.log1p(df["mileage"].clip(lower=0))

        if "vehicle_age" in df.columns:
            df["km_per_year"] = df["mileage"] / (df["vehicle_age"] + 1)
            df["km_per_year"] = df["km_per_year"].replace([np.inf, -np.inf], 0)

        return df

    # =========================
    # 3. Engine Features
    # =========================

    def _create_engine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "horsepower" in df.columns and "engine_size" in df.columns:
            df["engine_power_ratio"] = df["horsepower"] / (df["engine_size"] + 1e-6)

        if "engine_size" in df.columns:
            df["engine_size_log"] = np.log1p(df["engine_size"].clip(lower=0))

        return df

    # =========================
    # 4. Brand Features
    # =========================

    def _create_brand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "make" not in df.columns:
            return df

        df["make"] = df["make"].astype(str).str.lower()
        df["luxury_brand"] = df["make"].isin(self.luxury_brands).astype(int)

        return df

    # =========================
    # 5. Market / Segment Features
    # =========================

    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "horsepower" not in df.columns:
            return df

        hp = df["horsepower"]
        df["premium_segment"] = (
            ((hp > 200) | (df.get("luxury_brand", 0) == 1))
            .astype(int)
        )

        return df

    # =========================
    # 6. Interaction Features (UPDATED: Added defensive safety checks)
    # =========================

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Captures nonlinear dependencies between variables defensively."""
        if "vehicle_age" in df.columns and "mileage" in df.columns:
            df["age_mileage_interaction"] = df["vehicle_age"] * df["mileage"]

        if "horsepower" in df.columns and "vehicle_age" in df.columns:
            df["power_depreciation"] = df["horsepower"] / (df["vehicle_age"] + 1)

        return df

    # =========================
    # 7. German Market Features
    # =========================

    def _create_german_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds realistic German market pricing signals and adjustments.
        Multipliers applied to base price to reflect real-world factors.
        All features are optional for backward compatibility.
        """

        df = df.copy()

        # A) Condition feature
        if "condition" in df.columns:
            condition_mapping = {
                "excellent": 1.15,
                "good": 1.0,
                "fair": 0.80,
            }
            df["condition_multiplier"] = (
                df["condition"]
                .astype(str)
                .str.lower()
                .map(condition_mapping)
                .fillna(1.0)
            )

        # B) Accident history
        if "accident" in df.columns:
            accident_mapping = {
                "yes": 0.88,
                "true": 0.88,
                "1": 0.88,
                "no": 1.0,
                "false": 1.0,
                "0": 1.0,
            }
            df["accident_multiplier"] = (
                df["accident"]
                .astype(str)
                .str.lower()
                .map(accident_mapping)
                .fillna(1.0)
            )

        # C) Service history
        if "service_history" in df.columns:
            service_mapping = {
                "full": 1.08,
                "partial": 1.0,
                "none": 0.90,
            }
            df["service_multiplier"] = (
                df["service_history"]
                .astype(str)
                .str.lower()
                .map(service_mapping)
                .fillna(1.0)
            )

        # D) Trim level feature
        if "trim" in df.columns:
            trim_mapping = {
                "base": 1.0,
                "sport": 1.05,
                "m_sport": 1.12,
                "amg": 1.15,
                "s_line": 1.10,
            }
            df["trim_multiplier"] = (
                df["trim"]
                .astype(str)
                .str.lower()
                .map(trim_mapping)
                .fillna(1.0)
            )

        # E) TÜV feature
        if "tuv_months" in df.columns:
            df["tuv_months"] = pd.to_numeric(df["tuv_months"], errors="coerce").fillna(6)
            df["tuv_multiplier"] = df["tuv_months"].apply(
                lambda x: 0.85 if x < 3 else (1.0 if x <= 12 else 1.05)
            )

        # F) Fuel efficiency feature
        if "consumption_l_per_100km" in df.columns:
            consumption = pd.to_numeric(df["consumption_l_per_100km"], errors="coerce")
            df["consumption_normalized"] = consumption.apply(
                lambda x: 1.0 / np.log1p(np.clip(x, a_min=0.1, a_max=50)) / 0.5
                if pd.notna(x)
                else 1.0
            )

        # G) Composite price multiplier combining all German market effects
        multiplier_cols = [
            "condition_multiplier",
            "accident_multiplier",
            "service_multiplier",
            "trim_multiplier",
            "tuv_multiplier",
        ]
        existing_multipliers = [c for c in multiplier_cols if c in df.columns]
        if existing_multipliers:
            df["market_adjustment"] = df[existing_multipliers].prod(axis=1)

        return df

    # =========================
    # 8. Nonlinear Transforms
    # =========================

    def _apply_nonlinear_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        if "mileage" in df.columns:
            df["sqrt_mileage"] = np.sqrt(df["mileage"].clip(lower=0))

        if "vehicle_age" in df.columns:
            df["inverse_age"] = 1 / (df["vehicle_age"] + 1)

        return df