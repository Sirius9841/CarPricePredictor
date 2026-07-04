from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from catboost import Pool

from src.utils import logger, regression_metrics


# =========================
# Evaluation Class
# =========================

class ModelEvaluator:
    """
    Production-grade evaluation toolkit for car price models.
    """

    def __init__(self, model, n_splits: int = 5):
        self.model = model
        self.n_splits = n_splits

    # =========================
    # Cross Validation
    # =========================

    def cross_validate(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            categorical_features: list[str],
    ) -> dict:
        """
        K-Fold cross validation with CatBoost.
        """

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        scores = {
            "MAE": [],
            "RMSE": [],
            "R2": [],
            "MAPE": [],
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pool = Pool(X_train, y_train, cat_features=categorical_features)
            val_pool = Pool(X_val, y_val, cat_features=categorical_features)

            model = self.model.__class__(**self.model.get_params())
            model.fit(train_pool, verbose=False)

            preds = model.predict(X_val)

            fold_metrics = regression_metrics(y_val, preds)

            for k in scores:
                scores[k].append(fold_metrics[k])

            logger.info(f"Fold {fold + 1} metrics: {fold_metrics}")

        summary = {k: float(np.mean(v)) for k, v in scores.items()}

        logger.info(f"CV Summary: {summary}")

        return summary

    # =========================
    # Residual Analysis
    # =========================

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Shows residual distribution.
        """

        residuals = y_true - y_pred

        plt.figure()
        plt.hist(residuals, bins=50)
        plt.title("Residual Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.show()

    # =========================
    # Prediction vs Actual
    # =========================

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Visualizes prediction quality.
        """

        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.4)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()])
        plt.title("Prediction vs Actual")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.show()

    # =========================
    # Error Breakdown by Segment
    # =========================

    def segment_error_analysis(
            self,
            df: pd.DataFrame,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            segment_col: str = "premium_segment",
    ) -> None:
        """
        Compares model error across different market segments.
        """

        df = df.copy()
        df["error"] = np.abs(y_true - y_pred)

        grouped = df.groupby(segment_col)["error"].mean()

        logger.info(f"Segment error analysis:\n{grouped}")

    # =========================
    # Feature Importance
    # =========================

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Extracts feature importance from CatBoost.
        """

        importance = self.model.get_feature_importance()

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values(by="importance", ascending=False)