#!/usr/bin/env python3
"""
Run Finnhub news fetch using the same stocks and time frames as the Yahoo caller.

Usage:
    python run_finnhub_news.py

Loads configs/run.yaml for universe.tickers, start_date, end_date, and (if present)
data/interim/peertickers.json for peer tickers — matching scripts/02_fetch_news.py.
Writes normalized news (from Finnhub API, all sources including Yahoo) to
data/raw/news/yahoo_news.jsonl.
Requires FINNHUB_API_KEY in the environment.
"""

import json
import sys
from pathlib import Path

from _bootstrap import *

from datetime import date, timedelta

from common.config import load_config
from common.paths import INTERIM_DATA_DIR, NEWS_RAW_DATA_DIR
from pipelines.finnhub_news_pipeline import fetch_and_save_finnhub_news


def _load_peer_tickers(tickers: list[str]) -> list[str]:
    """Load peer tickers from peertickers.json for first base ticker, if present."""
    path = INTERIM_DATA_DIR / "peertickers.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    first = (tickers or [""])[0]
    return data.get(first, [])


def main() -> int:
    run = load_config("run.yaml")
    tickers = run["universe"]["tickers"]
    if not tickers:
        print("No tickers in run.yaml universe.tickers", file=sys.stderr)
        return 1

    end_date = run["universe"].get("end_date") or date.today().isoformat()
    start_date = run["universe"].get("start_date") or (
        date.fromisoformat(end_date) - timedelta(days=7)
    ).isoformat()
    limit_per_ticker = int(run.get("news_limit_per_ticker", 200))

    peer_list = _load_peer_tickers(tickers)

    path, tickers_used = fetch_and_save_finnhub_news(
        tickers,
        peer_tickers=peer_list,
        start_date=start_date,
        end_date=end_date,
        limit_per_ticker=limit_per_ticker,
    )
    print("Saved:", path)
    print("Tickers used:", tickers_used)
    return 0


if __name__ == "__main__":
    sys.exit(main())
