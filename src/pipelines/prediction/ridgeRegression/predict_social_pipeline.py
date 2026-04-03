"""
predict_social_pipeline.py

Loads the saved social_vol_model.pkl and predicts volatility for a single
ticker using its latest available sentiment date as the prediction point.
Applies day-level weighting during feature construction. Windows where any
day has num_posts == 0 are rejected.
"""

import os
import pickle

import numpy as np

from pipelines.prediction.ridgeRegression.ridgeCreateModel_social_pipeline import (
    LOOKBACK,
    N_FEATURES,
    N_PER_DAY,
    ALL_FEATURES,
    load_json,
    _sorted_dates,
    _window_features,
    _window_has_posts,
    _ticker_mean_vol,
    _validate_weights,
)


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_model(model_dir: str, filename: str = "social_vol_model.pkl") -> dict:
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
    day_weights: list[float],
    target_date: str | None = None,             # kept for backwards compatibility; ignored
    feature_weights: list[float] | None = None, # kept for backwards compatibility; ignored
) -> tuple[np.ndarray, str]:
    """
    Build the feature row for a single ticker using its latest available
    sentiment date as the prediction point.

    Returns (X_row, date) or raises if the ticker cannot be evaluated.
    target_date and feature_weights are accepted for backwards compatibility
    but ignored.
    """
    if ticker not in sentiment_data:
        raise ValueError(f"Ticker '{ticker}' not found in sentiment data.")
    if ticker not in volatility_data:
        raise ValueError(f"Ticker '{ticker}' not found in volatility data.")

    sent      = sentiment_data[ticker]
    vol       = volatility_data[ticker]
    all_dates = _sorted_dates(sent)

    if len(all_dates) < LOOKBACK + 1:
        raise ValueError(f"Not enough sentiment days for '{ticker}' to form a window.")

    # latest date is the prediction target; everything before it is the window
    date        = all_dates[-1]
    prior_dates = all_dates[:-1]

    if not _window_has_posts(sent, prior_dates):
        raise ValueError(f"[{ticker}] Zero-post day in the window — cannot predict.")

    mean_vol = _ticker_mean_vol(vol)
    X_row    = _window_features(sent, vol, prior_dates, day_weights, vol_fill=mean_vol)

    return X_row.reshape(1, -1), date


# ── Entry point ───────────────────────────────────────────────────────────────

def run_prediction_pipeline(
    sentiment_path: str,
    volatility_path: str,
    model_dir: str,
    ticker: str,
    day_weights: list[float],
    target_date: str | None = None,             # kept for backwards compatibility; ignored
    feature_weights: list[float] | None = None, # kept for backwards compatibility; ignored
) -> dict:
    """
    Load the saved model and predict volatility for a single ticker using its
    latest available sentiment date.

    target_date and feature_weights are accepted for backwards compatibility
    but ignored.

    Returns
    -------
    {
        "ticker"    : ticker symbol,
        "date"      : date predicted for (latest available),
        "predicted" : predicted volatility (float),
    }
    """
    _validate_weights(day_weights, feature_weights)

    sentiment_data  = load_json(sentiment_path)
    volatility_data = load_json(volatility_path)
    model_bundle    = load_model(model_dir)

    X, date = build_eval_row(
        sentiment_data, volatility_data, ticker, day_weights
    )

    X_scaled = model_bundle["scaler"].transform(X)
    y_pred   = float(model_bundle["model"].predict(X_scaled)[0])

    print(f"[{ticker}] {date}  |  predicted: {y_pred:.6f}")

    return {
        "ticker"   : ticker,
        "date"     : date,
        "predicted": y_pred,
    }