from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from src.utils import logger, clip_outliers


# =========================
# Main Preprocessing Class
# =========================

class CarDataPreprocessor:
    """
    Production-grade preprocessing pipeline for used car price prediction.
    Designed for real-world messy datasets (German market).
    """

    def __init__(
            self,
            target_col: str = "price",
            categorical_features: Optional[List[str]] = None,
            numerical_features: Optional[List[str]] = None,
    ):
        self.target_col = target_col
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []

        self.fitted = False

    # =========================
    # Public API
    # =========================

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full training preprocessing pipeline.
        """
        df = self._clean(df)
        df = self._handle_missing(df)
        df = self._normalize_categoricals(df)
        df = self._handle_outliers(df)
        self.fitted = True

        logger.info("Preprocessing fit_transform completed.")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inference preprocessing pipeline.
        Must behave identically to training.
        """
        if not self.fitted:
            logger.warning("Preprocessor not fitted. Running transform anyway.")

        df = self._clean(df)
        df = self._handle_missing(df)
        df = self._normalize_categoricals(df)

        logger.info("Preprocessing transform completed.")
        return df

    # =========================
    # Cleaning
    # =========================

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning:
        - remove duplicates
        - remove invalid rows
        """

        initial_shape = df.shape

        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Remove impossible values
        if "price" in df.columns:
            df = df[df["price"] > 100]  # filter junk listings
            df = df[df["price"] < 2_000_000]

        if "mileage" in df.columns:
            df = df[df["mileage"] >= 0]

        if "year" in df.columns:
            df = df[(df["year"] >= 1980) & (df["year"] <= 2026)]

        logger.info(f"Cleaned data: {initial_shape} → {df.shape}")
        return df

    # =========================
    # Missing Values
    # =========================

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent missing value handling.
        """
        df = df.copy()

        for col in df.columns:
            # Check if the column is a numeric data type (int or float)
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                df[col] = df[col].fillna(median)
            else:
                # Fallback for strings, categoricals, booleans, etc.
                df[col] = df[col].fillna("unknown")

        return df

    # =========================
    # Categorical Normalization
    # =========================

    def _normalize_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes categorical text fields.
        """

        df = df.copy()

        for col in df.select_dtypes(include="object").columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace({"none": "unknown", "nan": "unknown"})
            )

        return df

    # =========================
    # Outlier Handling
    # =========================

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip numerical outliers instead of dropping data.
        """

        df = df.copy()

        for col in df.select_dtypes(include=[np.number]).columns:

            if col == self.target_col:
                continue

            df = clip_outliers(df, col)

        return df

    # =========================
    # Feature Separation
    # =========================

    def split_features_target(
            self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        return X, y