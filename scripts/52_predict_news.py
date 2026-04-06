from _bootstrap import *

import json
import os
from datetime import datetime

from common.config import load_config
from pipelines.prediction.ridgeRegression.predict_news_pipeline import run_prediction_pipeline


# ── Paths ─────────────────────────────────────────────────────────────────────

SENTIMENT_PATH  = "data/processed/news/news_data.json"
VOLATILITY_PATH = "data/processed/market/volatilitySingleTicker.json"
MODEL_DIR       = "artifacts/models"
OUTPUT_PATH     = "artifacts/predictions/news_prediction.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_predictions(path: str) -> dict:
    """Load existing predictions file, or return empty structure."""
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {path}, starting fresh.")
    return {}


def save_predictions(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def insert_prediction(store: dict, ticker: str, date: str, result: dict):
    run_meta = {
        "run": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    entry = {**result, **run_meta}

    store.setdefault(ticker, {}).setdefault(date, [])

    run_number = len(store[ticker][date]) + 1
    entry["run"] = run_number

    # newest run first
    store[ticker][date].insert(0, entry)

    if run_number > 1:
        print(
            f"  Note: run #{run_number} added for {ticker} on {date} "
            f"(previous run{'s' if run_number > 2 else ''} preserved)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run = load_config("run.yaml")

    tickers = run["universe"]["ticker_symbols"]
    day_weights = run["day_weights"]
    target_date = run["universe"]["target_date"]
    feature_weights_news = run["feature_weights_news"]

    if not tickers:
        raise ValueError("No ticker symbols found in config/run.yaml")

    store = load_predictions(OUTPUT_PATH)

    for ticker in tickers:
        print(f"Ticker          : {ticker}")

        result = run_prediction_pipeline(
            sentiment_path=SENTIMENT_PATH,
            volatility_path=VOLATILITY_PATH,
            model_dir=MODEL_DIR,
            ticker=ticker,
            target_date=target_date,
            day_weights=day_weights,
            feature_weights=feature_weights_news,
        )

        date = result["date"]
        print(f"Target date     : {date}")

        insert_prediction(store, ticker, date, result)
        save_predictions(OUTPUT_PATH, store)
        print(f"Result saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()