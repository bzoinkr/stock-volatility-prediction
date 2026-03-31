from _bootstrap import *

import json

from common.config import load_config
from pipelines.prediction.ridgeRegression.eval_news_pipeline import run_eval_pipeline


NEWS_PATH = "data/processed/news/news_data_train.json"
VOLATILITY_PATH = "data/processed/market/volatilitySingleTicker.json"
MODEL_DIR = "artifacts/models"


def main():
    run = load_config("run.yaml")

    ticker = run["universe"]["ticker_symbols"][0]
    start_eval_date = run["universe"]["start_date"]
    end_eval_date = run["universe"]["target_date"]
    day_weights = run["day_weights"]
    feature_weights = run["feature_weights_news"]

    print(f"Evaluating {ticker} from {start_eval_date} to {end_eval_date}")

    results = run_eval_pipeline(
        news_path=NEWS_PATH,
        volatility_path=VOLATILITY_PATH,
        model_dir=MODEL_DIR,
        ticker=ticker,
        start_eval_date=str(start_eval_date),
        end_eval_date=str(end_eval_date),
        day_weights=day_weights,
        feature_weights=feature_weights,
    )

    output_path = "artifacts/evaluations/news_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()