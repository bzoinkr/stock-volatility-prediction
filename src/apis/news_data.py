from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import yfinance as yf


def fetch_yahoo_news(
    ticker: str,
    start_date: str | None = None,   # "YYYY-MM-DD"
    end_date: str | None = None,     # "YYYY-MM-DD"
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """
    Fetch Yahoo Finance news for a ticker between [start_date, end_date] inclusive.

    Defaults:
        - end_date = today (local date)
        - start_date = end_date - 7 days

    Returns:
        List of normalized dicts with:
        ticker, date, published_at (UTC ISO), title, link, publisher, summary
    """
    t = ticker.upper().strip()

    # --------------------------------------------------
    # Resolve date window
    # --------------------------------------------------
    end = date.today() if end_date is None else date.fromisoformat(end_date)
    start = (end - timedelta(days=7)) if start_date is None else date.fromisoformat(start_date)
    if start > end:
        start, end = end, start

    # --------------------------------------------------
    # Pull raw Yahoo news
    # --------------------------------------------------
    raw_items: list[dict[str, Any]] = []
    try:
        raw_items = yf.Ticker(t).news or []
    except Exception:
        raw_items = []

    # --------------------------------------------------
    # Normalize, filter, and collect
    # --------------------------------------------------
    out: list[dict[str, Any]] = []
    for it in raw_items[: max(limit * 3, 300)]:  # buffer in case many are out-of-range
        content = it.get("content") or {}

        # Yahoo provides ISO timestamps like "2026-02-05T18:30:00Z"
        pub_iso = content.get("pubDate")
        if pub_iso is None:
            continue

        try:
            published = datetime.fromisoformat(pub_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue

        d = published.date()
        if not (start <= d <= end):
            continue

        # Prefer canonical URL, fall back to click-through
        canonical = (content.get("canonicalUrl") or {}).get("url")
        click = (content.get("clickThroughUrl") or {}).get("url")
        link = canonical or click

        publisher = (content.get("provider") or {}).get("displayName")

        out.append(
            {
                "ticker": t,
                "date": d.isoformat(),
                "published_at": published.isoformat(),
                "title": content.get("title"),
                "link": link,
                "publisher": publisher,
                "summary": content.get("summary") or content.get("description"),
            }
        )

        if len(out) >= limit:
            break

    return out
