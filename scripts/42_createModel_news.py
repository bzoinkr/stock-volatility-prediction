from _bootstrap import *

import json

from common.config import load_config
from pipelines.prediction.ridgeRegression.ridgeCreateModel_news_pipeline import run_pipeline


NEWS_PATH = "data/processed/news/news_data_train.json"
VOLATILITY_PATH = "data/processed/market/volatilitySingleTicker.json"
MODEL_DIR = "artifacts/models"


def main():
    run = load_config("run.yaml")
    ticker = run["universe"]["ticker_symbols"][0]
    date = run["universe"]["target_date"]
    day_weights = run["day_weights"]
    feature_weights = run["feature_weights_news"]

    print(f"Predicting {ticker} volatility for date  : {date}")

    predictions = run_pipeline(
        news_path=NEWS_PATH,
        volatility_path=VOLATILITY_PATH,
        model_dir=MODEL_DIR,
        ticker=ticker,
        target_date=str(date),
        day_weights=day_weights,
        feature_weights=feature_weights,
    )

    output_path = "data/predictions/news_predictions.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()