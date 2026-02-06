# imports data from yahoo finance & using pandas
import yfinance as yf
import pandas as pd

# Usually YahooFinance news items are dictionaries / safe way to finding info without crashing
def _nested(d, *keys, default=None):
    """Safe nested dict getter."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# Finds the time of publish / if no time, returns None to prevent errors
def _parse_time(it):
    """
    Tries common yfinance news time fields.
    Returns pandas Timestamp (UTC) or NaT.
    """
    ts = (
        it.get("providerPublishTime")
        or it.get("providerPublishTime")  # (in case your version uses this spelling)
        or _nested(it, "content", "providerPublishTime")
        or _nested(it, "content", "pubDate")
        or _nested(it, "content", "pubDateTime")
    )

    if ts is None:
        return pd.NaT

    # epoch seconds
    if isinstance(ts, (int, float)):
        return pd.to_datetime(ts, unit="s", utc=True)

    # ISO-ish string
    return pd.to_datetime(ts, utc=True, errors="coerce")


# The main function to pull raw info of the news and turns the data into tabluar
def yahoo_news(count=50, tab="news", days_back=None):
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

    df = pd.DataFrame(rows)

    if days_back is not None:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        df = df[df["published_utc"].notna() & (df["published_utc"] >= cutoff)]

    df = df.sort_values("published_utc", ascending=False).drop_duplicates(subset=["link", "title"])
    df = df.reset_index(drop=True)

    df.insert(0, "id", df.index + 1)

    return df


# This function runs if the file is ran directly
if __name__ == "__main__":
    df = yahoo_news(count=50, tab="news", days_back=14)

    if df.empty:
        print("No news returned (try tab='all' or increase count).")
    else:
        print(df[["id", "published_utc", "publisher", "title", "link"]].head(10).to_string(index=False))
        df.to_csv("UNH_yahoo_news.csv", index=False)
        print("\nSaved: UNH_yahoo_news.csv")
