"""
src/model.py — XGBoost Signal Prediction Model
================================================

This module wraps an XGBoost classifier that predicts the next-candle
direction (BUY / SELL / HOLD) from a feature-enriched OHLCV DataFrame.
It loads a pre-trained model serialised by train_model.py (via joblib),
runs inference on the latest feature row, and returns a prediction with
a confidence score.  It also exposes helpers for offline training and
model persistence so that train_model.py can delegate to this class.

Usage:
    from src.model import SignalModel
    model = SignalModel()
    model.load("models/xgb_signal_model.json")
    result = model.predict(enriched_df)
    # result → {"signal": "BUY", "confidence": 0.82}
"""

import pandas as pd


# Mapping from integer label to human-readable signal name
LABEL_MAP: dict[int, str] = {0: "SELL", 1: "HOLD", 2: "BUY"}


class SignalModel:
    """
    Wrapper around an XGBoost classifier for directional signal prediction.

    Attributes
    ----------
    model : xgboost.XGBClassifier or None
        The underlying trained model instance.
    feature_columns : list[str] or None
        Ordered list of feature column names the model expects.
    """

    def __init__(self) -> None:
        """Initialise with no model loaded."""
        # TODO: Set up model and feature_columns attributes
        pass

    def load(self, path: str) -> None:
        """
        Load a previously saved XGBoost model from *path* using joblib.

        Parameters
        ----------
        path : str
            Path to the saved model file (.joblib or .json).

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        # TODO: Load model from disk via joblib.load
        pass

    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Run inference on the latest (last) row of *features_df*.

        Parameters
        ----------
        features_df : pd.DataFrame
            Must contain all columns listed in self.feature_columns.

        Returns
        -------
        dict
            {"signal": "BUY"|"SELL"|"HOLD",
             "confidence": float,
             "probabilities": dict[str, float]}

        Raises
        ------
        RuntimeError
            If no model has been loaded yet.
        """
        # TODO: Extract feature row, call predict_proba, map to label
        pass

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on every row of *features_df* (used for back-testing).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with appended columns: predicted_signal,
            confidence.
        """
        # TODO: Vectorised batch prediction
        pass

    def train(self, X: pd.DataFrame, y: pd.Series,
              params: dict | None = None) -> None:
        """
        Train (or retrain) the XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame   Feature matrix.
        y : pd.Series      Target labels (0=SELL, 1=HOLD, 2=BUY).
        params : dict       Optional XGBClassifier hyper-parameters override.
        """
        # TODO: Initialise XGBClassifier with params and call .fit()
        pass

    def save(self, path: str) -> None:
        """
        Persist the trained model to *path* using joblib.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        # TODO: Save model to disk via joblib.dump
        pass
