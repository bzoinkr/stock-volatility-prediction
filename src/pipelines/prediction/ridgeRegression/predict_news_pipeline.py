"""
eval_social_pipeline.py

Loads the saved social_vol_model.pkl and evaluates it on a single date (end_date)
for a single ticker.
Applies the same feature-level and day-level weighting used during training.
Missing lagged volatility values in the window are filled with the ticker mean.
"""

import os
import pickle

import numpy as np

from pipelines.prediction.ridgeRegression.ridgeCreateModel_news_pipeline import (
    LOOKBACK,
    N_FEATURES,
    N_PER_DAY,
    ALL_FEATURES,
    load_json,
    _sorted_dates,
    _window_features,
    _ticker_mean_vol,
    _validate_weights,
)


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_model(model_dir: str, filename: str = "news_vol_model.pkl") -> dict:
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Dataset construction ──────────────────────────────────────────────────────

def build_eval_row(
    sentiment_data: dict,
    volatility_data: dict,
    ticker: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> np.ndarray:
    """
    Build the weighted feature row for a single ticker on target_date.
    Returns (X_row, y_true) or (None, None) if the ticker cannot be evaluated.
    """
    if ticker not in sentiment_data:
        raise ValueError(f"Ticker '{ticker}' not found in sentiment data.")
    if ticker not in volatility_data:
        raise ValueError(f"Ticker '{ticker}' not found in volatility data.")

    sent = sentiment_data[ticker]
    vol  = volatility_data[ticker]

    prior_dates = [d for d in _sorted_dates(sent) if d < target_date]
    if len(prior_dates) < 1:
        raise ValueError(f"No prior sentiment days available for '{ticker}' before {target_date}.")

    mean_vol = _ticker_mean_vol(vol)
    X_row    = _window_features(sent, vol, prior_dates, day_weights, feature_weights, vol_fill=mean_vol)

    return X_row.reshape(1, -1)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_prediction_pipeline(
    sentiment_path: str,
    volatility_path: str,
    model_dir: str,
    ticker: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> dict:
    """
    Load the saved model and predict volatility for a single ticker on target_date.

    Returns
    -------
    {
        "ticker"    : ticker symbol,
        "date"      : target date,
        "predicted" : predicted volatility (float),
        "actual"    : actual volatility (float),
    }
    """
    _validate_weights(day_weights, feature_weights)

    sentiment_data  = load_json(sentiment_path)
    volatility_data = load_json(volatility_path)
    model_bundle    = load_model(model_dir)

    X = build_eval_row(
        sentiment_data, volatility_data, ticker, target_date, day_weights, feature_weights
    )

    X_scaled  = model_bundle["scaler"].transform(X)
    y_pred    = float(model_bundle["model"].predict(X_scaled)[0])

    print(f"[{ticker}] {target_date}  |  predicted: {y_pred:.6f}")

    return {
        "ticker"   : ticker,
        "date"     : target_date,
        "predicted": y_pred,
    }