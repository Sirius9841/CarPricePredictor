from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from functools import wraps
from typing import Callable, Any, Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Logging Setup
# =========================

def get_logger(name: str = "car_price_predictor") -> logging.Logger:
    """
    Creates a standardized logger for the project.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


logger = get_logger()


# =========================
# Timing Decorator
# =========================

def timer(func: Callable) -> Callable:
    """
    Measures execution time of functions (training, preprocessing, etc.).
    """

    import time

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        logger.info(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result

    return wrapper


# =========================
# Metrics
# =========================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    Handles division stability issues.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def regression_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Unified metric output for evaluation pipeline.
    """

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


# =========================
# Data Validation Helpers
# =========================

def check_dataframe(df: pd.DataFrame, required_cols: list[str]) -> None:
    """
    Validates dataset integrity before training.
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        logger.warning(f"Missing columns detected: {missing}")


def safe_log1p(x: pd.Series) -> pd.Series:
    """
    Safe log transform that handles negative/zero values.
    """
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def clip_outliers(
        df: pd.DataFrame,
        col: str,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Clips extreme outliers instead of removing data (better for ML stability).
    """

    lower = df[col].quantile(lower_quantile)
    upper = df[col].quantile(upper_quantile)

    df[col] = df[col].clip(lower, upper)
    return df


# =========================
# Train/Test Utilities
# =========================

def train_test_split_time_aware(
        df: pd.DataFrame,
        time_col: str,
        test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optional time-aware split for vehicle age bias reduction.
    """

    df = df.sort_values(time_col)
    split_idx = int(len(df) * (1 - test_size))

    return df.iloc[:split_idx], df.iloc[split_idx:]