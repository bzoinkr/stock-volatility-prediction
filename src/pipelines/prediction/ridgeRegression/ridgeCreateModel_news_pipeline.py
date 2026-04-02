import json
import os
import pickle

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


LOOKBACK = 5

SENTIMENT_FEATURES = [
    "num_posts",
    "mean_compound",
    "mean_pos",
    "mean_neg",
    "mean_neu",
    "pos_neg_ratio",
    "sentiment_balance",
    "bullish_ratio",
    "bearish_ratio",
    "variance_compound",
    "std_compound",
]

ALL_FEATURES = SENTIMENT_FEATURES + ["volatility", "avg_weekly_vol", "avg_monthly_vol"]

N_PER_DAY = len(ALL_FEATURES)       # 14
N_FEATURES = LOOKBACK * N_PER_DAY   # 70


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_model(model_bundle: dict, model_dir: str, filename: str = "news_vol_model.pkl") -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(model_bundle, f)
    return path


def normalize_news_data(news_raw):
    if isinstance(news_raw, dict):
        return news_raw

    out = {}
    for row in news_raw:
        ticker = str(row["ticker"])
        date = str(row["date"])

        out.setdefault(ticker, {})
        out[ticker][date] = {
            "num_posts": row.get("num_posts", 0.0),
            "mean_compound": row.get("mean_compound", 0.0),
            "mean_pos": row.get("mean_pos", 0.0),
            "mean_neg": row.get("mean_neg", 0.0),
            "mean_neu": row.get("mean_neu", 0.0),
            "pos_neg_ratio": row.get("pos_neg_ratio", 0.0),
            "sentiment_balance": row.get("sentiment_balance", 0.0),
            "bullish_ratio": row.get("bullish_ratio", 0.0),
            "bearish_ratio": row.get("bearish_ratio", 0.0),
            "variance_compound": row.get("variance_compound", 0.0),
            "std_compound": row.get("std_compound", 0.0),
        }

    return out


def _sorted_dates(date_dict: dict) -> list[str]:
    return sorted(date_dict.keys())


def _rolling_avg_vol(volatility_ticker: dict, date: str, window: int) -> float:
    """
    Compute the mean of the most recent `window` daily vol values
    on dates strictly before `date`.  Returns 0.0 if no values exist.
    """
    sorted_dates = sorted(d for d in volatility_ticker if d < date)
    recent = sorted_dates[-window:]
    vals = [volatility_ticker[d] for d in recent if volatility_ticker[d] is not None]
    return float(np.mean(vals)) if vals else 0.0


def _feature_row(day_data: dict, vol_value: float, avg_weekly: float, avg_monthly: float, feature_weights: list[float]) -> np.ndarray:
    raw = [float(day_data.get(col, 0.0) or 0.0) for col in SENTIMENT_FEATURES]
    raw.append(float(vol_value) if vol_value is not None else 0.0)
    raw.append(float(avg_weekly))
    raw.append(float(avg_monthly))
    return np.array(raw) * np.array(feature_weights)


def _window_features(
    sentiment_ticker: dict,
    volatility_ticker: dict,
    dates_before: list[str],
    day_weights: list[float],
    feature_weights: list[float],
    fill_vol: float = 0.0,
) -> np.ndarray:
    window = dates_before[-LOOKBACK:]
    weights_oldest_first = list(reversed(day_weights))

    pad = LOOKBACK - len(window)
    window = ([""] * pad) + list(window)

    vec = []
    for day, dw in zip(window, weights_oldest_first):
        day_data = sentiment_ticker.get(day, {})
        vol_value = volatility_ticker.get(day, fill_vol) if day else fill_vol
        avg_weekly  = _rolling_avg_vol(volatility_ticker, day, 5) if day else 0.0
        avg_monthly = _rolling_avg_vol(volatility_ticker, day, 21) if day else 0.0
        row = _feature_row(day_data, vol_value, avg_weekly, avg_monthly, feature_weights)
        vec.append(row * dw)

    return np.concatenate(vec)


def build_dataset(
    sentiment_data: dict,
    volatility_data: dict,
    ticker: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    X_rows, y_vals = [], []

    if ticker not in volatility_data:
        raise ValueError(f"{ticker} is not in volatilitySingleTicker.json")

    if ticker not in sentiment_data:
        raise ValueError(f"{ticker} is not in news_data_train.json")

    sentiment = sentiment_data[ticker]
    vol = volatility_data[ticker]
    dates = _sorted_dates(sentiment)

    for i, date in enumerate(dates):
        if date >= target_date:
            break
        if date not in vol:
            continue

        prior_dates = dates[:i]

        X_rows.append(
            _window_features(
                sentiment,
                vol,
                prior_dates,
                day_weights,
                feature_weights,
                fill_vol=0.0,
            )
        )
        y_vals.append(float(vol[date]))

    if not X_rows:
        return np.empty((0, N_FEATURES)), np.array([])

    return np.array(X_rows), np.array(y_vals)


def build_prediction_row(
    sentiment_data: dict,
    volatility_data: dict,
    ticker: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> np.ndarray:
    if ticker not in volatility_data:
        raise ValueError(f"{ticker} is not in volatilitySingleTicker.json")

    if ticker not in sentiment_data:
        raise ValueError(f"{ticker} is not in news_data_train.json")

    sentiment = sentiment_data[ticker]
    vol = volatility_data[ticker]

    prior_dates = [d for d in _sorted_dates(sentiment) if d < target_date]
    if len(prior_dates) < 1:
        raise ValueError(f"No prior dates available for {ticker} before {target_date}")

    X_pred = _window_features(
        sentiment,
        vol,
        prior_dates,
        day_weights,
        feature_weights,
        fill_vol=0.0,
    )

    return X_pred.reshape(1, -1)


def compute_feature_target_correlations(
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict]:
    X_recent = X[:, -N_PER_DAY:]

    results = []
    for i, name in enumerate(ALL_FEATURES):
        col = X_recent[:, i]
        if np.std(col) == 0:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(col, y)

        results.append({
            "feature": name,
            "pearson_r": round(float(r), 6),
            "abs_r": round(abs(float(r)), 6),
            "p_value": round(float(p), 6),
            "significant": bool(p < 0.05),
        })

    results.sort(key=lambda x: x["abs_r"], reverse=True)
    return results


def print_correlations(correlations: list[dict]) -> None:
    print("\n  Feature -> Volatility Correlations (most recent day, sorted by |r|):")
    print(f"  {'Feature':<35}  {'Pearson r':>10}  {'p-value':>10}  {'Sig?':>6}")
    print("  " + "-" * 68)
    for c in correlations:
        sig = "yes" if c["significant"] else "no"
        print(f"  {c['feature']:<35}  {c['pearson_r']:>10.6f}  {c['p_value']:>10.6f}  {sig:>6}")


def save_correlations(correlations: list[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "news_feature_correlations.json")
    with open(path, "w") as f:
        json.dump(correlations, f, indent=4)
    return path


def train_model(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    scaler = StandardScaler(with_std=False)
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    return {"scaler": scaler, "model": model}


def predict(model_bundle: dict, X: np.ndarray) -> np.ndarray:
    X_scaled = model_bundle["scaler"].transform(X)
    return model_bundle["model"].predict(X_scaled)


def _validate_weights(day_weights: list[float], feature_weights: list[float]) -> None:
    if len(day_weights) != LOOKBACK:
        raise ValueError(f"day_weights must have {LOOKBACK} elements, got {len(day_weights)}.")
    if len(feature_weights) != N_PER_DAY:
        raise ValueError(f"feature_weights must have {N_PER_DAY} elements, got {len(feature_weights)}.")


def run_pipeline(
    news_path: str,
    volatility_path: str,
    model_dir: str,
    ticker: str,
    target_date: str,
    day_weights: list[float],
    feature_weights: list[float],
) -> dict[str, float]:
    _validate_weights(day_weights, feature_weights)

    news_raw = load_json(news_path)
    sentiment_data = normalize_news_data(news_raw)
    volatility_data = load_json(volatility_path)

    X, y = build_dataset(
        sentiment_data,
        volatility_data,
        ticker,
        target_date,
        day_weights,
        feature_weights,
    )

    if X.shape[0] < 2:
        raise ValueError(
            f"Only {X.shape[0]} training samples found for {ticker}. "
            f"You need more historical dates before {target_date}."
        )

    print(f"Training on {X.shape[0]} samples ({X.shape[1]} features each) for {ticker}.")

    correlations = compute_feature_target_correlations(X, y)
    print_correlations(correlations)
    corr_path = save_correlations(correlations, model_dir)
    print(f"\nCorrelations saved -> {corr_path}")

    model_bundle = train_model(X, y)
    saved_path = save_model(model_bundle, model_dir)
    print(f"Model saved -> {saved_path}")

    X_pred = build_prediction_row(
        sentiment_data,
        volatility_data,
        ticker,
        target_date,
        day_weights,
        feature_weights,
    )

    pred = float(predict(model_bundle, X_pred)[0])
    return {ticker: pred}