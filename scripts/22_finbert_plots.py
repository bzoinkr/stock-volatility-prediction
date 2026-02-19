from __future__ import annotations

from _bootstrap import *

from common.config import load_config
from evaluation.sentiment_plots import make_sentiment_report

if __name__ == "__main__":
    make_sentiment_report(
        input_jsonl="data/processed/news/yahoo_news_finbert_scored.jsonl",
        out_dir="artifacts/reports/sentiment/finbert",
        date_col="date",
        score_col="compound",
        group_col="ticker",
        rolling_window_days=5,
        verbose=True,
    )
