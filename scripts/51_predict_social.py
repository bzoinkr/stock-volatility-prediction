from _bootstrap import *

import json

from common.config import load_config
from pipelines.prediction.ridgeRegression.predict_social_pipeline import run_prediction_pipeline


# ── Paths ─────────────────────────────────────────────────────────────────────

SENTIMENT_PATH  = "data/processed/social/social_data.json"
VOLATILITY_PATH = "data/processed/market/volatility.json"
MODEL_DIR       = "artifacts/models"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    # Base tickers defined in the config universe
    TICKER = run["universe"]["ticker_symbols"][0]

    if not TICKER:
        raise ValueError("TICKER is not set. Please assign a ticker symbol in config/run.yaml")

    date            = run["universe"]["target_date"]
    day_weights     = run["day_weights"]
    feature_weights = run["feature_weights"]

    print(f"Ticker          : {TICKER}")
    print(f"Target date     : {date}")

    result = run_prediction_pipeline(
        sentiment_path  = SENTIMENT_PATH,
        volatility_path = VOLATILITY_PATH,
        model_dir       = MODEL_DIR,
        ticker          = TICKER,
        target_date     = str(date),
        day_weights     = day_weights,
        feature_weights = feature_weights,
    )

    output_path = f"artifacts/predictions/social_prediction.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()