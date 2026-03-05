from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import finnhub

# Chunk size in days for paginating the year (API returns limited results per request).
# Use 7-day chunks so we get full coverage; 30-day chunks only return the tail of each
# period (e.g. last week of each month), not the entire year.
CHUNK_DAYS = 7


def _fetch_finnhub_news_chunked(
    ticker: str,
    start: date,
    end: date,
    api_key: str,
) -> list[dict[str, Any]]:
    """Request company_news in small (week-sized) chunks and merge for full-year coverage."""
    seen: set[tuple[str, int]] = set()  # (url or headline, datetime) for dedupe
    merged: list[dict[str, Any]] = []
    client = finnhub.Client(api_key=api_key)
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end)
        try:
            items = client.company_news(
                ticker,
                _from=current.isoformat(),
                to=chunk_end.isoformat(),
            ) or []
            for it in items:
                ts = it.get("datetime")
                key = (it.get("url") or it.get("headline") or "", ts or 0)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(it)
        except Exception:
            pass
        current = chunk_end + timedelta(days=1)
        if current <= end:
            time.sleep(0.25)  # gentle rate limit between chunks
    # Sort by datetime descending (newest first)
    merged.sort(key=lambda x: x.get("datetime") or 0, reverse=True)
    return merged


def fetch_finnhub_news(
    ticker: str,
    start_date: str | None = None,   # "YYYY-MM-DD"
    end_date: str | None = None,     # "YYYY-MM-DD"
    *,
    limit: int = 10000,
    api_key: str | None = None,
    exclude_sources: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch Finnhub company news for a ticker between [start_date, end_date] inclusive.

    Defaults:
        - end_date = today (local date)
        - start_date = end_date - 1 year

    API key: set FINNHUB_API_KEY env var or pass api_key=.

    exclude_sources: drop items whose source (publisher) matches any of these
        strings, case-insensitive (e.g. ("Yahoo",) to exclude Yahoo).

    Returns:
        List of normalized dicts with:
        ticker, date, published_at (UTC ISO), title, link, publisher, summary
    """
    t = ticker.upper().strip()

    # --------------------------------------------------
    # Resolve date window
    # --------------------------------------------------
    end = date.today() if end_date is None else date.fromisoformat(end_date)
    start = (end - timedelta(days=365)) if start_date is None else date.fromisoformat(start_date)
    if start > end:
        start, end = end, start

    # --------------------------------------------------
    # Setup client and pull raw Finnhub news (chunked by week to get full year)
    # --------------------------------------------------
    key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise ValueError("Finnhub API key required: set FINNHUB_API_KEY or pass api_key=")

    raw_items = _fetch_finnhub_news_chunked(t, start, end, key)

    # --------------------------------------------------
    # Normalize, filter, and collect
    # --------------------------------------------------
    excluded = {s.strip().lower() for s in (exclude_sources or ())}
    out: list[dict[str, Any]] = []
    for it in raw_items[: max(limit * 3, 300)]:  # buffer in case many are out-of-range
        if excluded and (it.get("source") or "").strip().lower() in excluded:
            continue
        # Finnhub: datetime is Unix timestamp (seconds), headline, url, source, summary
        ts = it.get("datetime")
        if ts is None:
            continue

        try:
            published = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            continue

        d = published.date()
        if not (start <= d <= end):
            continue

        out.append(
            {
                "ticker": t,
                "date": d.isoformat(),
                "published_at": published.isoformat(),
                "title": it.get("headline"),
                "link": it.get("url"),
                "publisher": it.get("source"),
                "summary": it.get("summary"),
            }
        )

        if len(out) >= limit:
            break

    return out


def fetch_finnhub_news_raw(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch raw company news from Finnhub (no normalization).

    Same date window and API key as fetch_finnhub_news. Returns the exact
    list of dicts from Finnhub (category, datetime, headline, id, image,
    related, source, summary, url). Use this to inspect why results include
    mixed topics: Finnhub returns news "related" to the symbol (sector/theme),
    not only articles solely about the ticker.
    """
    t = ticker.upper().strip()
    end = date.today() if end_date is None else date.fromisoformat(end_date)
    start = (end - timedelta(days=365)) if start_date is None else date.fromisoformat(start_date)
    if start > end:
        start, end = end, start

    key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise ValueError("Finnhub API key required: set FINNHUB_API_KEY or pass api_key=")

    try:
        return _fetch_finnhub_news_chunked(t, start, end, key)
    except Exception:
        return []
