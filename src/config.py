from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ModelConfig:
    """
    Central configuration for model training and inference.
    """

    target: str = "price"
    log_target: bool = True

    random_state: int = 42

    test_size: float = 0.2

    cv_folds: int = 5

    n_jobs: int = -1

    # Default model selection
    model_type: str = "catboost"  # catboost | lightgbm | xgboost


@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for dataset structure and feature expectations.
    """

    categorical_features: List[str] = (
        "make",
        "model",
        "trim",
        "fuel",
        "transmission",
        "drive_type",
        "color",
        "city",
        "seller_type",
        "accident",
        "service_history",
        "condition",
    )

    numerical_features: List[str] = (
        "year",
        "mileage",
        "engine_size",
        "horsepower",
        "owners",
        "doors",
        "tuv_months",
        "consumption_l_per_100km",
    )

    optional_features: List[str] = (
        "equipment",
    )


@dataclass(frozen=True)
class FeatureConfig:
    """
    Feature engineering configuration.
    """

    enable_age_features: bool = True
    enable_log_features: bool = True
    enable_ratio_features: bool = True
    enable_luxury_features: bool = True

    luxury_brands: List[str] = (
        "bmw",
        "mercedes-benz",
        "audi",
        "porsche",
        "lexus",
        "jaguar",
        "land rover",
        "tesla",
    )


@dataclass(frozen=True)
class OptunaConfig:
    """
    Hyperparameter tuning configuration.
    """

    n_trials: int = 50
    timeout: Optional[int] = None

    cv_folds: int = 5
    direction: str = "minimize"  # for RMSE


@dataclass(frozen=True)
class PathConfig:
    """
    File system paths.
    """

    data_path: str = "data/car_data.csv"
    model_dir: str = "models/"
    report_dir: str = "reports/"