import json
import os
from datetime import date, timedelta

from apis.finnhub_news import fetch_finnhub_news
from common.paths import NEWS_RAW_DATA_DIR


# --------------------------------------------------
# Fetch and save Finnhub news for tickers in a date range
# --------------------------------------------------
def fetch_and_save_finnhub_news(
    base_tickers: list[str],
    peer_tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    limit_per_ticker: int = 200,
    filename: str | None = None,
    exclude_sources: tuple[str, ...] | None = None,
    api_key: str | None = None,
) -> tuple[str, list[str]]:
    """
    Fetch Finnhub news for base tickers + peer tickers and save to the raw news directory.

    Uses the same ticker list and date window as the Yahoo news pipeline.
    All sources (including Yahoo) are included unless exclude_sources is set.

    Returns:
        (output_path, tickers_used)
    """
    # Defaults: same as Yahoo caller — last 7 days
    end_date = end_date or date.today().isoformat()
    start_date = start_date or (date.fromisoformat(end_date) - timedelta(days=7)).isoformat()

    # Normalize + de-dupe while preserving order (same as Yahoo pipeline)
    seen = set()
    tickers_used = []
    for t in base_tickers + peer_tickers:
        t = t.upper()
        if t not in seen:
            seen.add(t)
            tickers_used.append(t)

    # Output location
    out_dir = NEWS_RAW_DATA_DIR
    os.makedirs(NEWS_RAW_DATA_DIR, exist_ok=True)

    if filename is None:
        filename = "yahoo_news.jsonl"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        for t in tickers_used:
            rows = fetch_finnhub_news(
                t,
                start_date=start_date,
                end_date=end_date,
                limit=limit_per_ticker,
                api_key=api_key,
                exclude_sources=exclude_sources,
            )
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_path, tickers_used
