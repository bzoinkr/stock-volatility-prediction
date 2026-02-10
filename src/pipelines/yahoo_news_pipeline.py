import json
import os
from datetime import date, timedelta

from apis.news_data import fetch_yahoo_news
from apis.ollama_data import get_peer_tickers
from common.paths import NEWS_RAW_DATA_DIR


# --------------------------------------------------
# Expand base tickers using Ollama peer tickers
# --------------------------------------------------
def expand_tickers_with_peers(base_tickers: list[str], peer_k: int = 6) -> list[str]:
    """Return a de-duplicated list of base tickers plus Ollama-derived peer tickers."""
    seen = set()
    out = []

    for t in base_tickers:
        t = t.upper()

        if t not in seen:
            seen.add(t)
            out.append(t)

        for p in get_peer_tickers(t, k=peer_k):
            p = p.upper()
            if p not in seen:
                seen.add(p)
                out.append(p)

    return out


# --------------------------------------------------
# Fetch and save Yahoo news for tickers in a date range
# --------------------------------------------------
def fetch_and_save_yahoo_news(
    base_tickers: list[str],
    peer_tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    limit_per_ticker: int = 200,
    filename: str | None = None,
) -> tuple[str, list[str]]:
    """
    Fetch Yahoo news for base tickers + peer tickers and save to the raw news directory.

    Returns:
        (output_path, tickers_used)
    """
    # Defaults: last 7 days
    end_date = end_date or date.today().isoformat()
    start_date = start_date or (date.fromisoformat(end_date) - timedelta(days=7)).isoformat()

    # Normalize + de-dupe while preserving order
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

    # Default filename includes the date window for traceability
    if filename is None:
        filename = f"yahoo_news_{start_date}_{end_date}.jsonl"
    out_path = os.path.join(out_dir, filename)

    # Fetch per ticker and write one JSON object per line (JSONL)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in tickers_used:
            rows = fetch_yahoo_news(
                t,
                start_date=start_date,
                end_date=end_date,
                limit=limit_per_ticker,
            )
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_path, tickers_used
