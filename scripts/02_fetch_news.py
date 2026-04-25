from _bootstrap import *

import os
from datetime import date, timedelta

from common.config import load_config
from pipelines.news.finnhub_news_pipeline import fetch_and_save_finnhub_news


def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    # By default, fetch for a single target ticker.
    # For batch orchestrators, allow comma-separated override via env.
    override = os.environ.get("NEWS_TICKERS")
    if override:
        tickers = [t.strip().upper() for t in override.split(",") if t.strip()]
    else:
        tickers = [run["universe"]["ticker_symbols"][0]]

    # Fail fast if no tickers were found
    if not tickers:
        raise KeyError(
            f"No tickers found in run.yaml. Available keys: {list(run.keys())}\n"
            "Add one of these keys: tickers / symbols / stocks"
        )

    # --------------------------------------------------
    # Fetch news for configured ticker(s) only (saved to yahoo_news_train.jsonl)
    # --------------------------------------------------
    # Pull a full rolling year of news ending on configured end_date (or today).
    end_date = run["universe"]["end_date"] or date.today().isoformat()
    start_date = (date.fromisoformat(end_date) - timedelta(days=365)).isoformat()
    limit_per_ticker = int(run.get("news_limit_per_ticker", 200))
    output_filename = os.environ.get("NEWS_OUTPUT_FILENAME", "yahoo_news_train.jsonl")
    append_mode = os.environ.get("NEWS_APPEND", "0") == "1"
    finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
    if not finnhub_api_key:
        raise ValueError("Missing FINNHUB_API_KEY. Add it to your .env file.")

    path, tickers_used = fetch_and_save_finnhub_news(
        tickers,
        start_date=start_date,
        end_date=end_date,
        limit_per_ticker=limit_per_ticker,
        filename=output_filename,
        append=append_mode,
        api_key=finnhub_api_key,
    )
    print("Saved:", path)
    print("Tickers used:", tickers_used)


if __name__ == "__main__":
    main()
