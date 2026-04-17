"""
pipeline.py

Volatility prediction pipeline.

Each (ticker, date) pair is one training sample. The feature vector is the
concatenation of LOOKBACK=5 windows, each window containing:
    - 15 sentiment features
    - 1 lagged volatility value  (0.0 if missing for that day)
= 16 features per day × 5 days = 80 features total.

Day weights (length-5, index 0 = most recent) are applied via sign-preserving
exponentiation (sign(x) * |x|^w) during feature construction.

One shared HuberRegressor model is trained across all tickers and saved as a
single .pkl file.
"""

import json
import os
import pickle

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


# ── Config ───────────────────────────────────────────────────────────────────

LOOKBACK = 5

SENTIMENT_FEATURES = [
    "num_posts",
    "log_impressions",
    "mean_compound",
    "impression_weighted_compound",
    "mean_pos",
    "mean_neg",
    "mean_neu",
    "pos_neg_ratio",
    "sentiment_balance",
    "bullish_ratio",
    "bearish_ratio",
    "variance_compound",
    "std_compound",
    "term_diversity",
    "top_term_ratio",
]

ALL_FEATURES = SENTIMENT_FEATURES + ["volatility"]
N_PER_DAY    = len(ALL_FEATURES)        # 16
N_FEATURES   = LOOKBACK * N_PER_DAY    # 80


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_model(model_bundle: dict, model_dir: str, filename: str = "social_vol_model.pkl") -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(model_bundle, f)
    return path


# ── Feature helpers ───────────────────────────────────────────────────────────

def _sorted_dates(date_dict: dict) -> list[str]:
    return sorted(date_dict.keys())


def _signed_pow(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Sign-preserving power: sign(x) * |x|^w

    This handles negative feature values safely when w is fractional or
    non-integer.  Special cases:
      - w == 0  → 0.0  (suppress the feature rather than returning x^0 = 1)
      - x == 0  → 0.0  (0^w = 0 for any positive w)
    """
    signs  = np.sign(x)
    result = signs * (np.abs(x) ** w)
    result = np.where(w == 0.0, 0.0, result)   # weight=0 suppresses feature
    return result


def _feature_row(day_data: dict, vol_value: float) -> np.ndarray:
    """
    Single day → raw feature vector of length 16.
    Sentiment nulls become 0.0. Missing volatility becomes 0.0.
    """
    raw = [float(day_data.get(col) or 0.0) for col in SENTIMENT_FEATURES]
    raw.append(float(vol_value) if vol_value is not None else 0.0)
    return np.array(raw)


def _ticker_mean_vol(volatility_ticker: dict) -> float:
    """Mean of all available volatility values for a ticker; 0.0 if none."""
    vals = [v for v in volatility_ticker.values() if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def _window_features(
    sentiment_ticker: dict,
    volatility_ticker: dict,
    dates_before: list[str],
    day_weights: list[float],
    vol_fill: float = 0.0,
) -> np.ndarray:
    """
    Flatten the last LOOKBACK days from dates_before into one weighted row.

    dates_before is sorted ascending (oldest first), so weights are reversed:
        window[0] = oldest → day_weights[-1]
        window[-1] = newest → day_weights[0]

    Day weight is applied as a power via sign-preserving exponentiation.

    If the window has fewer than LOOKBACK days, pad the front with empty rows.
    Missing volatility values are filled with vol_fill (ticker mean vol).
    """
    window               = dates_before[-LOOKBACK:]
    weights_oldest_first = list(reversed(day_weights))

    # pad front with empty sentinel days if window shorter than LOOKBACK
    pad    = LOOKBACK - len(window)
    window = ([""] * pad) + list(window)

    vec = []
    for day, dw in zip(window, weights_oldest_first):
        sent_row  = sentiment_ticker.get(day, {})
        vol_value = volatility_ticker.get(day, vol_fill) if day else vol_fill
        row       = _feature_row(sent_row, vol_value)
        vec.append(_signed_pow(row, np.full(row.shape, dw)))

    return np.concatenate(vec)


def _window_has_posts(
    sentiment_ticker: dict,
    dates_before: list[str],
) -> bool:
    """
    Returns False if any day in the last LOOKBACK window has num_posts == 0
    (or missing). Padded days (empty string sentinel) are also treated as
    zero-post days and will cause the window to be rejected.
    """
    window = dates_before[-LOOKBACK:]
    # padded days count as zero-post
    if len(window) < LOOKBACK:
        return False
    for day in window:
        posts = sentiment_ticker.get(day, {}).get("num_posts", 0) or 0
        if posts == 0:
            return False
    return True


# ── Dataset construction ──────────────────────────────────────────────────────

def build_dataset(
    sentiment_data: dict,
    volatility_data: dict,
    target_date: str | None = None,   # kept for backwards compatibility; ignored
    day_weights: list[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cross-ticker (X, y) matrices using ALL available dates.

    A date is included as a training sample if:
      - it has at least LOOKBACK prior sentiment days for that ticker
      - a volatility label exists for that date (missing = weekend/holiday, skipped)
      - every day in the LOOKBACK window has num_posts > 0

    target_date is accepted for backwards compatibility but ignored.
    Missing lagged volatility values within the window are filled with the
    ticker's mean volatility rather than dropping the sample entirely.
    """
    if day_weights is None:
        raise ValueError("day_weights must be provided.")

    X_rows, y_vals = [], []
    tickers = sorted(set(sentiment_data) & set(volatility_data))

    for ticker in tickers:
        sent  = sentiment_data[ticker]
        vol   = volatility_data[ticker]
        dates = _sorted_dates(sent)

        mean_vol = _ticker_mean_vol(vol)

        for i, date in enumerate(dates):
            if i < LOOKBACK:
                # need at least LOOKBACK prior days to form a full window
                continue
            if date not in vol:
                # no volatility label for this date (weekend/holiday) — skip
                continue

            prior_dates = dates[:i]
            if not _window_has_posts(sent, prior_dates):
                continue
            X_rows.append(_window_features(sent, vol, prior_dates, day_weights, vol_fill=mean_vol))
            y_vals.append(float(vol[date]))

    if not X_rows:
        return np.empty((0, N_FEATURES)), np.array([])

    return np.array(X_rows), np.array(y_vals)


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_feature_target_correlations(
    X: np.ndarray,
    y: np.ndarray,
    day_weights: list[float],
) -> list[dict]:
    """
    Compute Pearson correlation of each feature (most-recent day's block,
    unweighted) against the volatility target y.
    Sorted by |correlation| descending.
    """
    X_recent = X[:, -N_PER_DAY:]

    results = []
    for i, name in enumerate(ALL_FEATURES):
        col = X_recent[:, i]
        if np.std(col) == 0:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(col, y)
        results.append({
            "feature"    : name,
            "pearson_r"  : round(float(r), 6),
            "abs_r"      : round(abs(float(r)), 6),
            "p_value"    : round(float(p), 6),
            "significant": bool(p < 0.05),
        })

    results.sort(key=lambda x: x["abs_r"], reverse=True)
    return results


def print_correlations(correlations: list[dict]) -> None:
    print("\n  Feature → Volatility Correlations (most recent day, sorted by |r|):")
    print(f"  {'Feature':<35}  {'Pearson r':>10}  {'p-value':>10}  {'Sig?':>6}")
    print("  " + "-" * 68)
    for c in correlations:
        sig = "yes" if c["significant"] else "no"
        print(f"  {c['feature']:<35}  {c['pearson_r']:>10.6f}  {c['p_value']:>10.6f}  {sig:>6}")


def save_correlations(correlations: list[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "social_feature_correlations.json")
    with open(path, "w") as f:
        json.dump(correlations, f, indent=4)
    return path


# ── Model ─────────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    # Use with_std=False: only center (remove mean), do NOT divide by std.
    # Dividing by std would erase the variance differences introduced by
    # day_weights, making them ineffective. Centering alone removes constant
    # offsets while preserving the relative scaling the weights impose.
    scaler   = StandardScaler(with_std=False)
    X_scaled = scaler.fit_transform(X)
    model    = HuberRegressor(alpha=alpha, epsilon=1.5, max_iter=2000)
    model.fit(X_scaled, y)
    n_outliers = int(np.sum(model.outliers_))
    print(f"HuberRegressor: {n_outliers}/{len(y)} samples flagged as outliers "
          f"(alpha={alpha}, epsilon={model.epsilon})")
    return {"scaler": scaler, "model": model}


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_weights(day_weights: list[float], feature_weights: list[float] | None = None) -> None:
    if len(day_weights) != LOOKBACK:
        raise ValueError(f"day_weights must have {LOOKBACK} elements, got {len(day_weights)}.")
    # feature_weights is accepted for backwards compatibility but ignored.


# ── Entry point ───────────────────────────────────────────────────────────────

def run_pipeline(
    sentiment_path: str,
    volatility_path: str,
    model_dir: str,
    target_date: str | None = None,               # kept for backwards compatibility; ignored
    day_weights: list[float] = None,
    feature_weights: list[float] | None = None,   # kept for backwards compatibility; ignored
    alpha: float = 1.0,
    l1_ratio: float | None = None,                # kept for backwards compatibility; ignored
) -> None:
    """
    1. Load data
    2. Build cross-ticker training dataset using ALL available dates per ticker
       (any date with >= LOOKBACK prior sentiment days, a volatility label,
       and num_posts > 0 for every day in the window)
    3. Compute and print feature→target correlations
    4. Train one HuberRegressor model, save to model_dir/social_vol_model.pkl

    target_date, feature_weights, and l1_ratio are accepted for backwards
    compatibility but ignored.
    """
    _validate_weights(day_weights, feature_weights)

    sentiment_data  = load_json(sentiment_path)
    volatility_data = load_json(volatility_path)

    # ── build dataset ─────────────────────────────────────────────────────
    X, y = build_dataset(sentiment_data, volatility_data, day_weights=day_weights)

    if X.shape[0] < 2:
        raise ValueError(f"Only {X.shape[0]} training samples — need at least 2.")

    print(f"Training on {X.shape[0]} samples ({X.shape[1]} features each).")
    print(f"Day weights (newest→oldest) : {day_weights}")

    # ── correlations ──────────────────────────────────────────────────────
    correlations = compute_feature_target_correlations(X, y, day_weights)
    print_correlations(correlations)
    corr_path = save_correlations(correlations, model_dir)
    print(f"\nCorrelations saved → {corr_path}")

    # ── train & save ──────────────────────────────────────────────────────
    model_bundle = train_model(X, y, alpha=alpha)
    saved_path   = save_model(model_bundle, model_dir)
    print(f"Model saved    → {saved_path}")