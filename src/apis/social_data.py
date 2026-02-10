from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

import requests

DEFAULT_USER_AGENT = "stock-volatility-prediction/1.0 (social_data)"


@dataclass(frozen=True)
class RedditListingParams:
    limit: int = 100
    max_pages: int = 5
    sleep_s: float = 1.2
    include_over18: bool = False
    base: str = "https://old.reddit.com"   # old is more tolerant


def _get_json(url: str, *, user_agent: str = DEFAULT_USER_AGENT, timeout_s: int = 30) -> Dict[str, Any]:
    resp = requests.get(url, headers={"User-Agent": user_agent, "Accept": "application/json"}, timeout=timeout_s)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        snippet = resp.text[:400].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response from Reddit. First 400 chars: {snippet}")


def _utc_date_from_created_utc(created_utc: float) -> str:
    dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
    return dt.date().isoformat()


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_submission(child: Dict[str, Any]) -> Dict[str, Any]:
    data = child.get("data", {}) if isinstance(child, dict) else {}

    post_id = data.get("id")
    fullname = data.get("name")
    created_utc = float(data.get("created_utc", 0.0) or 0.0)

    title = _clean_text(data.get("title", "") or "")
    selftext = _clean_text(data.get("selftext", "") or "")
    subreddit = data.get("subreddit", "") or ""
    author = data.get("author", "") or ""

    permalink = data.get("permalink", "") or ""
    url = data.get("url", "") or ""

    over18 = bool(data.get("over_18", False))
    combined = (title + "\n" + selftext).strip()

    return {
        "platform": "reddit",
        "type": "submission",
        "id": post_id,
        "fullname": fullname,
        "subreddit": subreddit,
        "author": author,
        "created_utc": created_utc,
        "date_utc": _utc_date_from_created_utc(created_utc) if created_utc else None,
        "title": title,
        "selftext": selftext,
        "text": _clean_text(combined),
        "score": int(data.get("score", 0) or 0),
        "num_comments": int(data.get("num_comments", 0) or 0),
        "permalink": f"https://www.reddit.com{permalink}" if permalink.startswith("/") else permalink,
        "post_url": url,
        "over_18": over18,
    }


def build_subreddit_new_url(
    subreddit: str,
    *,
    base: str = "https://old.reddit.com",
    limit: int = 100,
    after: Optional[str] = None,
) -> str:
    params = {"limit": limit, "raw_json": 1}
    if after:
        params["after"] = after
    return f"{base}/r/{subreddit}/new.json?{urlencode(params)}"


def fetch_subreddit_new(
    subreddit: str,
    *,
    params: Optional[RedditListingParams] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    return_scrape_urls: bool = False,
) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], List[str]]:
    """
    Fetch newest submissions from a subreddit via /new.json (NOT search).
    Returns normalized posts and (optionally) the URLs scraped.
    """
    p = params or RedditListingParams()
    results: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    scrape_urls: List[str] = []

    after = None
    pages = 0

    while pages < p.max_pages:
        url = build_subreddit_new_url(subreddit, base=p.base, limit=p.limit, after=after)
        scrape_urls.append(url)

        data = _get_json(url, user_agent=user_agent)
        listing = data.get("data", {})
        children = listing.get("children", []) or []

        for child in children:
            norm = normalize_submission(child)
            pid = norm.get("id")
            if not pid:
                continue
            if norm.get("over_18") and not p.include_over18:
                continue
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            results.append(norm)

        after = listing.get("after")
        pages += 1
        if not after:
            break
        time.sleep(p.sleep_s)

    if return_scrape_urls:
        return results, scrape_urls
    return results


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dedupe_rows(rows: List[Dict[str, Any]], *, key: str = "id") -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        k = r.get(key)
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out
