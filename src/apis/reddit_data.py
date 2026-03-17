from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlencode

import requests


PAGE_SIZE = 100  # Reddit's hard maximum per request


def fetch_posts_for_ticker(
    ticker: str,
    keywords: List[str],
    *,
    subreddits: List[str],
    max_posts: int,
    session: requests.Session,
) -> List[Dict[str, Any]]:
    terms = [ticker, f"${ticker}"] + keywords
    escaped = [re.escape(t) for t in terms if t]
    rx = re.compile(r"(" + r"|".join(escaped) + r")", flags=re.IGNORECASE)

    sub_part = " OR ".join(f"subreddit:{s}" for s in subreddits)
    term_part = " OR ".join(terms)
    query = f"({sub_part}) ({term_part})"

    base_params = {"q": query, "sort": "new", "t": "all", "limit": PAGE_SIZE, "raw_json": 1}
    base_url = "https://old.reddit.com/search.json"

    posts = []
    after = None

    while len(posts) < max_posts:
        params = {**base_params}
        if after:
            params["after"] = after

        resp = session.get(f"{base_url}?{urlencode(params)}", timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {})

        children = data.get("children", [])
        if not children:
            break  # no more results

        for ch in children:
            d = ch.get("data", {})
            text = ((d.get("title") or "") + "\n" + (d.get("selftext") or "")).strip()

            m = rx.search(text)
            if not m:
                continue

            posts.append({
                "id": d.get("id"),
                "ticker": ticker,
                "matched_term": m.group(1),
                "subreddit": d.get("subreddit"),
                "created_utc": d.get("created_utc"),
                "text": text,
                "score": d.get("score"),
                "num_comments": d.get("num_comments"),
                "permalink": d.get("permalink"),
            })

        after = data.get("after")
        if not after:
            break  # Reddit has no more pages

    return posts