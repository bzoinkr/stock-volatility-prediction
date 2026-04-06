"""
predict_social_pipeline.py

Loads the saved social_vol_model.pkl and predicts volatility for a single
ticker using either a specified target_date or its latest available sentiment
date as the prediction point. Applies day-level weighting during feature
construction. Windows where any day has num_posts == 0 are rejected.
"""

import os
import pickle

import numpy as np

from pipelines.prediction.ridgeRegression.huberCreateModel_social_pipeline import (
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
    target_date: str | None = None,
    feature_weights: list[float] | None = None,  # kept for backwards compatibility; ignored
) -> tuple[np.ndarray, str]:
    """
    Build the feature row for a single ticker.

    If target_date is provided, the LOOKBACK days immediately preceding it in
    the sorted sentiment dates are used as the window, and the prediction is
    made for target_date.

    If target_date is None, falls back to the original behaviour: the latest
    available sentiment date is the prediction target and everything before it
    is the window.

    Returns (X_row, date) or raises if the ticker cannot be evaluated.
    """
    if ticker not in sentiment_data:
        raise ValueError(f"Ticker '{ticker}' not found in sentiment data.")
    if ticker not in volatility_data:
        raise ValueError(f"Ticker '{ticker}' not found in volatility data.")

    sent      = sentiment_data[ticker]
    vol       = volatility_data[ticker]
    all_dates = _sorted_dates(sent)

    if target_date is not None:
        if target_date not in sent:
            raise ValueError(
                f"[{ticker}] target_date '{target_date}' not found in sentiment data."
            )

        target_idx = all_dates.index(target_date)

        if target_idx < LOOKBACK:
            raise ValueError(
                f"[{ticker}] Not enough prior sentiment days before '{target_date}' "
                f"to form a {LOOKBACK}-day window (need {LOOKBACK}, have {target_idx})."
            )

        date        = target_date
        prior_dates = all_dates[target_idx - LOOKBACK : target_idx]

    else:
        # Original behaviour: predict for the latest available date
        if len(all_dates) < LOOKBACK + 1:
            raise ValueError(f"Not enough sentiment days for '{ticker}' to form a window.")

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
    target_date: str | None = None,
    feature_weights: list[float] | None = None,  # kept for backwards compatibility; ignored
) -> dict | None:
    """
    Load the saved model and predict volatility for a single ticker.

    If target_date is provided, predicts for that specific date using the
    LOOKBACK days before it as the feature window.
    If target_date is None, predicts for the latest available sentiment date.

    Returns
    -------
    {
        "ticker"    : ticker symbol,
        "date"      : date predicted for,
        "predicted" : predicted volatility (float),
    }
    or None if the ticker cannot be evaluated (logged to stdout).
    """
    _validate_weights(day_weights, feature_weights)

    sentiment_data  = load_json(sentiment_path)
    volatility_data = load_json(volatility_path)
    model_bundle    = load_model(model_dir)

    try:
        X, date = build_eval_row(
            sentiment_data, volatility_data, ticker, day_weights,
            target_date=target_date,
        )
    except ValueError as e:
        print(f"  Skipped [{ticker}]: {e}")
        return None

    X_scaled = model_bundle["scaler"].transform(X)
    y_pred   = float(model_bundle["model"].predict(X_scaled)[0])

    print(f"[{ticker}] {date}  |  predicted: {y_pred:.6f}")

    return {
        "ticker"   : ticker,
        "date"     : date,
        "predicted": y_pred,
    }