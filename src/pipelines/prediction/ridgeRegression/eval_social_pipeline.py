"""
eval_social_pipeline.py

Loads the saved vol_model.pkl and evaluates it on a single date (end_date).
Applies the same feature-level and day-level weighting used during training.
Missing lagged volatility values in the window are filled with 0.0.
"""

import json
import os
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.prediction.ridgeRegression.ridgeCreateModel_social_pipeline import (
    LOOKBACK,
    N_FEATURES,
    N_PER_DAY,
    ALL_FEATURES,
    load_json,
    _sorted_dates,
    _window_features,
    _ticker_mean_vol,
    compute_feature_target_correlations,
    print_correlations,
    save_correlations,
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

def build_eval_rows(
    sentiment_data: dict,
    volatility_data: dict,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> tuple[np.ndarray, list[float], list[str]]:
    """
    For each ticker, build the weighted feature row for target_date and look
    up the actual volatility label on that date (if available).
    Missing lagged volatility values in the window are filled with 0.0.
    """
    X_rows, y_true, tickers_out = [], [], []

    tickers = sorted(set(sentiment_data) & set(volatility_data))

    for ticker in tickers:
        sent = sentiment_data[ticker]
        vol  = volatility_data[ticker]

        if target_date not in vol:
            print(f"[{ticker}] No volatility label for {target_date}, skipping.")
            continue

        mean_vol    = _ticker_mean_vol(vol)
        prior_dates = [d for d in _sorted_dates(sent) if d < target_date]
        if len(prior_dates) < 1:
            print(f"[{ticker}] No prior sentiment days available, skipping.")
            continue

        X_rows.append(_window_features(sent, vol, prior_dates, day_weights, feature_weights, vol_fill=mean_vol))
        y_true.append(float(vol[target_date]))
        tickers_out.append(ticker)

    if not X_rows:
        return np.empty((0, N_FEATURES)), [], []

    return np.array(X_rows), y_true, tickers_out


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals  = y_pred - y_true
    abs_errors = np.abs(residuals)
    qlike = lambda y_true, y_pred: np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)    
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    mape_vals    = abs_errors / np.where(y_true != 0, y_true, np.nan)
    mape         = float(np.nanmean(mape_vals)) * 100
    mean_vol     = float(np.mean(y_true))
    baseline_mae = float(np.mean(np.abs(y_true - mean_vol)))

    return {
        "n_tickers"   : len(y_true),
        "mae"         : float(mae),
        "rmse"        : float(rmse),
        "r2"          : float(r2),
        "mape_pct"    : float(mape),
        "max_error"   : float(np.max(abs_errors)),
        "min_error"   : float(np.min(abs_errors)),
        "mean_error"  : float(np.mean(residuals)),
        "std_error"   : float(np.std(residuals)),
        "baseline_mae": baseline_mae,
        "skill_score" : float(1 - mae / baseline_mae) if baseline_mae > 0 else float("nan"),
        "qlike"       : float(qlike)
    }


def print_stats(
    stats: dict,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> None:
    print("\n" + "=" * 52)
    print(f"  Evaluation on {target_date}  ({stats['n_tickers']} tickers)")
    print(f"  Day weights     (newest→oldest) : {day_weights}")
    print(f"  Feature weights                 : {feature_weights}")
    print("=" * 52)
    print(f"  MAE             : {stats['mae']:.6f}")
    print(f"  qlike             : {stats['qlike']:.6f}")
    print(f"  RMSE            : {stats['rmse']:.6f}")
    print(f"  R²              : {stats['r2']:.4f}")
    print(f"  MAPE            : {stats['mape_pct']:.2f}%")
    print(f"  Mean error(bias): {stats['mean_error']:.6f}")
    print(f"  Std of errors   : {stats['std_error']:.6f}")
    print(f"  Max abs error   : {stats['max_error']:.6f}")
    print(f"  Min abs error   : {stats['min_error']:.6f}")
    print("-" * 52)
    print(f"  Baseline MAE    : {stats['baseline_mae']:.6f}  (mean-vol naive)")
    skill = stats["skill_score"]
    label = "better" if skill > 0 else "worse"
    print(f"  Skill score     : {skill:.4f}  ({label} than baseline)")
    print("=" * 52)
    
# ── Entry point ───────────────────────────────────────────────────────────────

def run_eval_pipeline(
    sentiment_path: str,
    volatility_path: str,
    model_dir: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> dict:
    """
    Load the saved model, predict on target_date, compare against known
    volatility labels, and print evaluation statistics + correlations.

    Returns
    -------
    {
        "predictions" : {ticker: predicted_vol},
        "actuals"     : {ticker: actual_vol},
        "stats"       : aggregated error metrics,
        "correlations": feature→target correlation list,
    }
    """
    _validate_weights(day_weights, feature_weights)

    sentiment_data  = load_json(sentiment_path)
    volatility_data = load_json(volatility_path)
    model_bundle    = load_model(model_dir)

    print(f"Loaded model from : {os.path.join(model_dir, 'vol_model.pkl')}")

    # ── build eval rows ───────────────────────────────────────────────────
    X, y_true, tickers = build_eval_rows(
        sentiment_data, volatility_data, target_date, day_weights, feature_weights
    )

    if len(tickers) == 0:
        raise ValueError("No tickers had both a sentiment window and a volatility label.")

    y_true_arr = np.array(y_true)

    # ── correlations on eval set ──────────────────────────────────────────
    correlations = compute_feature_target_correlations(X, y_true_arr, day_weights)

    # ── predict ───────────────────────────────────────────────────────────
    X_scaled = model_bundle["scaler"].transform(X)
    y_pred   = model_bundle["model"].predict(X_scaled)

    # ── stats & printing ──────────────────────────────────────────────────
    stats = compute_stats(y_true_arr, y_pred)
    print_stats(stats, target_date, day_weights, feature_weights)

    return {
        "predictions" : dict(zip(tickers, y_pred.tolist())),
        "actuals"     : dict(zip(tickers, y_true)),
        "stats"       : stats,
        "correlations": correlations,
    }
