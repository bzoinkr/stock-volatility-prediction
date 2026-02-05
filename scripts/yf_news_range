import yfinance as yf
import pandas as pd

def _nested(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _parse_time(it):
    ts = (
        it.get("providerPublishTime")
        or _nested(it, "content", "providerPublishTime")
        or _nested(it, "content", "pubDate")
        or _nested(it, "content", "pubDateTime")
    )
    if ts is None:
        return pd.NaT
    if isinstance(ts, (int, float)):
        return pd.to_datetime(ts, unit="s", utc=True)
    return pd.to_datetime(ts, utc=True, errors="coerce")

def _to_utc(ts, tz):
    """Parse ts and convert to UTC. If ts has no timezone, assume tz."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(tz)
    return t.tz_convert("UTC")

def yahoo_news(count=500, tab="news", start_date=None, end_date=None, tz="America/New_York"):
    items = yf.Ticker("UNH").get_news(count=count, tab=tab)
    if not items:
        return pd.DataFrame()

    rows = []
    for it in items:
        title = it.get("title") or _nested(it, "content", "title")
        publisher = it.get("publisher") or _nested(it, "content", "provider", "displayName")
        link = (
            it.get("link")
            or _nested(it, "content", "canonicalUrl", "url")
            or _nested(it, "content", "clickThroughUrl", "url")
        )
        related = it.get("relatedTickers") or _nested(it, "content", "finance", "tickers") or []
        if isinstance(related, list):
            related = ",".join(related)

        rows.append({
            "published_utc": _parse_time(it),
            "publisher": publisher,
            "title": title,
            "link": link,
            "relatedTickers": related,
            "uuid": it.get("uuid") or _nested(it, "content", "id"),
        })

    df = pd.DataFrame(rows).dropna(subset=["published_utc"])

    # ---- Date range filtering ----
    if start_date is not None or end_date is not None:
        if start_date is not None:
            start_utc = _to_utc(start_date, tz)
            df = df[df["published_utc"] >= start_utc]

        if end_date is not None:
            # If end_date is given as "YYYY-MM-DD" (no time), make it inclusive by using next day as exclusive bound
            end_str = str(end_date)
            if len(end_str) == 10:  # looks like YYYY-MM-DD
                end_exclusive_utc = _to_utc(end_date, tz) + pd.Timedelta(days=1)
            else:
                end_exclusive_utc = _to_utc(end_date, tz)
            df = df[df["published_utc"] < end_exclusive_utc]

    # Sort newest first + dedupe + add id column
    df = df.sort_values("published_utc", ascending=False).drop_duplicates(subset=["link", "title"]).reset_index(drop=True)
    df.insert(0, "id", df.index + 1)

    return df

if __name__ == "__main__":
    # Example 1: exactly one day (year-month-day in New York time)
    df = yahoo_news(start_date="2026-01-30", end_date="2026-01-30")

    # Example 2: date range
    # df = yahoo_news_unh(start_date="2026-01-15", end_date="2026-01-31")

    # Example 3: time range within a day (NY time)
    # df = yahoo_news_unh(start_date="2026-02-01 09:30", end_date="2026-02-01 16:00")

    print(df[["id", "published_utc", "publisher", "title", "link"]].head(10).to_string(index=False))
    df.to_csv("UNH_yahoo_news2.csv", index=False)
    print("\nSaved: UNH_yahoo_news2.csv")
