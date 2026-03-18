from _bootstrap import *

from pathlib import Path
from common.config import load_config
from pipelines.news.sentiment_stats import build_sentiment_stats


def main():
    run = load_config("run.yaml")

    INPUT_PATH = Path("data/processed/news/yahoo_news_finbert_scored.jsonl")
    OUTPUT_PATH = Path("data/processed/news/news_data_train.json")

    START_DATE = run["universe"]["start_date"]
    END_DATE = run["universe"]["end_date"]

    build_sentiment_stats(INPUT_PATH, OUTPUT_PATH, START_DATE, END_DATE)


if __name__ == "__main__":
    main()