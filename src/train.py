from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional

from sklearn.model_selection import train_test_split, KFold

from catboost import CatBoostRegressor, Pool

from src.config import ModelConfig, PathConfig, FeatureConfig
from src.utils import logger, timer, regression_metrics
from src.preprocessing import CarDataPreprocessor
from src.feature_engineering import FeatureEngineer


# =========================
# Trainer Class
# =========================

class CarPriceTrainer:
    """
    Full training pipeline for used car price prediction.
    """

    def __init__(
            self,
            model_config: ModelConfig,
            path_config: PathConfig,
            feature_config: FeatureConfig,
    ):
        self.model_config = model_config
        self.path_config = path_config
        self.feature_config = feature_config

        self.model = None
        self.preprocessor = None
        self.feature_engineer = None

    # =========================
    # Load Data
    # =========================

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.path_config.data_path}")
        df = pd.read_csv(self.path_config.data_path)
        return df

    # =========================
    # Full Pipeline
    # =========================

    @timer
    def run(self) -> Dict[str, float]:

        df = self.load_data()

        # -------------------------
        # Preprocessing
        # -------------------------
        self.preprocessor = CarDataPreprocessor(
            target_col=self.model_config.target
        )

        df = self.preprocessor.fit_transform(df)

        # -------------------------
        # Feature Engineering
        # -------------------------
        self.feature_engineer = FeatureEngineer(
            luxury_brands=self.feature_config.luxury_brands
        )

        df = self.feature_engineer.fit_transform(df)

        # -------------------------
        # Split
        # -------------------------
        X, y = self.preprocessor.split_features_target(df)

        # log target
        if self.model_config.log_target:
            y = np.log1p(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
        )

        # -------------------------
        # Train Model
        # -------------------------
        self.model = self._train_catboost(X_train, y_train, X_val, y_val)

        # -------------------------
        # Evaluate
        # -------------------------
        preds = self.model.predict(X_val)

        if self.model_config.log_target:
            preds = np.expm1(preds)
            y_val = np.expm1(y_val)

        metrics = regression_metrics(y_val, preds)

        print("\n" + "=" * 65)
        print("        CAR PRICE PREDICTION - TRAINING REPORT")
        print("=" * 65)

        print("\nDataset")
        print("-" * 65)
        print(f"Rows used:            {len(df):,}")
        print(f"Features used:        {X.shape[1]}")
        print(f"Target:               log(price)")

        print("\nModel")
        print("-" * 65)
        print("Algorithm:            CatBoostRegressor")
        print(f"Best iteration:       {self.model.get_best_iteration()}")

        print("\nPerformance")
        print("-" * 65)
        print(f"MAE:                  €{metrics['MAE']:.2f}")
        print(f"RMSE:                 €{metrics['RMSE']:.2f}")
        print(f"R² Score:             {metrics['R2']:.4f}")
        print(f"MAPE:                 {metrics['MAPE']:.2f}%")

        print("\nInterpretation")
        print("-" * 65)
        print(f"• Average prediction error: about €{metrics['MAE']:.0f}")
        print(f"• Model explains {metrics['R2'] * 100:.2f}% of the price variation.")
        print(f"• Average percentage error: {metrics['MAPE']:.2f}%")

        print("\nSaved Artifacts")
        print("-" * 65)
        print("[OK] models/catboost_model.cbm")
        print("[OK] models/preprocessor.pkl")
        print("[OK] models/feature_engineer.pkl")

        print("=" * 65)
        print("Training completed successfully!")
        print("=" * 65)

        # -------------------------
        # Save Model
        # -------------------------
        self._save_model()
        importance = self.model.get_feature_importance()

        feature_importance = (
            pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            })
            .sort_values("Importance", ascending=False)
        )

        print("\nTop 10 Most Important Features")
        print("-" * 65)
        print(feature_importance.head(10).to_string(index=False))
        return metrics

    # =========================
    # CatBoost Training
    # =========================

    def _train_catboost(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
    ) -> CatBoostRegressor:

        categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        val_pool = Pool(X_val, y_val, cat_features=categorical_cols)

        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=self.model_config.random_state,
            verbose=False,
            early_stopping_rounds=100,
        )
        logger.info("Training CatBoost model...")

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )
        logger.info("Training completed.")
        logger.info(f"Best iteration: {model.get_best_iteration()}")
        return model

    # =========================
    # Save Model
    # =========================

    def _save_model(self) -> None:
        os.makedirs(self.path_config.model_dir, exist_ok=True)

        model_path = os.path.join(self.path_config.model_dir, "catboost_model.cbm")
        preprocessor_path = os.path.join(self.path_config.model_dir, "preprocessor.pkl")
        feature_engineer_path = os.path.join(self.path_config.model_dir, "feature_engineer.pkl")

        self.model.save_model(model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        joblib.dump(self.feature_engineer, feature_engineer_path)

        logger.info("Model and artifacts saved successfully")

if __name__ == "__main__":
    from src.config import ModelConfig, PathConfig, FeatureConfig

    trainer = CarPriceTrainer(
        model_config=ModelConfig(),
        path_config=PathConfig(),
        feature_config=FeatureConfig(),
    )

    metrics = trainer.run()
    print(metrics)