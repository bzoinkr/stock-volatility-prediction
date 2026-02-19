from __future__ import annotations

from _bootstrap import *

from common.config import load_config
from evaluation.sentiment_plots import make_sentiment_report

if __name__ == "__main__":
    make_sentiment_report(
        input_jsonl="data/processed/social/reddit_posts_vader_scored.jsonl",
        out_dir="artifacts/reports/sentiment/vader",
        date_col="date",
        score_col="compound",
        group_col="match_term",
        rolling_window_days=5,
        verbose=True,
    )
