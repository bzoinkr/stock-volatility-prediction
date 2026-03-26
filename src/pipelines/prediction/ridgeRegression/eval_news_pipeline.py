import json
import os
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.prediction.ridgeRegression.ridgeCreateModel_news_pipeline import (
    N_FEATURES,
    load_json,
    normalize_news_data,
    _sorted_dates,
    _window_features,
    compute_feature_target_correlations,
    _validate_weights,
)


def load_model(model_dir: str, filename: str = "news_vol_model.pkl") -> dict:
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_eval_rows_for_range(
    sentiment_data: dict,
    volatility_data: dict,
    ticker: str,
    start_eval_date: str,
    end_eval_date: str,
    day_weights: list[float],
    feature_weights: list[float],
):
    X_rows, y_true, eval_dates = [], [], []

    if ticker not in volatility_data:
        raise ValueError(f"{ticker} is not in volatilitySingleTicker.json")

    if ticker not in sentiment_data:
        raise ValueError(f"{ticker} is not in news_data_train.json")

    sentiment = sentiment_data[ticker]
    vol = volatility_data[ticker]

    sentiment_dates = _sorted_dates(sentiment)
    vol_dates = sorted(vol.keys())

    for date in vol_dates:
        if date < start_eval_date or date > end_eval_date:
            continue

        prior_dates = [d for d in sentiment_dates if d < date]
        if len(prior_dates) < 1:
            continue

        X_rows.append(
            _window_features(
                sentiment,
                vol,
                prior_dates,
                day_weights,
                feature_weights,
            )
        )
        y_true.append(float(vol[date]))
        eval_dates.append(date)

    if not X_rows:
        return np.empty((0, N_FEATURES)), [], []

    return np.array(X_rows), y_true, eval_dates


def compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    mean_vol = float(np.mean(y_true))
    baseline_mae = float(np.mean(np.abs(y_true - mean_vol)))

    return {
        "n_samples": len(y_true),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "baseline_mae": baseline_mae,
        "skill_score": float(1 - mae / baseline_mae) if baseline_mae > 0 else float("nan"),
        "mean_error": float(np.mean(residuals)),
        "std_error": float(np.std(residuals)),
        "max_error": float(np.max(abs_errors)),
        "min_error": float(np.min(abs_errors)),
    }


def run_eval_pipeline(
    news_path: str,
    volatility_path: str,
    model_dir: str,
    ticker: str,
    start_eval_date: str,
    end_eval_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> dict:
    _validate_weights(day_weights, feature_weights)

    news_raw = load_json(news_path)
    sentiment_data = normalize_news_data(news_raw)
    volatility_data = load_json(volatility_path)
    model_bundle = load_model(model_dir)

    X, y_true, eval_dates = build_eval_rows_for_range(
        sentiment_data,
        volatility_data,
        ticker,
        start_eval_date,
        end_eval_date,
        day_weights,
        feature_weights,
    )

    if len(eval_dates) == 0:
        raise ValueError(f"No usable evaluation rows found for {ticker} in the requested date range.")

    y_true_arr = np.array(y_true)
    X_scaled = model_bundle["scaler"].transform(X)
    y_pred = model_bundle["model"].predict(X_scaled)

    stats = compute_stats(y_true_arr, y_pred)
    correlations = compute_feature_target_correlations(X, y_true_arr)

    predictions = {d: float(p) for d, p in zip(eval_dates, y_pred.tolist())}
    actuals = {d: float(a) for d, a in zip(eval_dates, y_true)}

    return {
        "ticker": ticker,
        "eval_start_date": start_eval_date,
        "eval_end_date": end_eval_date,
        "predictions": predictions,
        "actuals": actuals,
        "stats": stats,
        "correlations": correlations,
    }