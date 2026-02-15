from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from apis.social_data import (
    RedditListingParams,
    dedupe_rows,
    fetch_subreddit_new,
    write_jsonl,
)


@dataclass
class RedditSocialPipelineConfig:
    subreddits: List[str]
    limit: int
    max_pages: int
    sleep_s: float
    include_over18: bool
    max_posts_per_ticker: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_keywords_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): [str(x) for x in (v or [])] for k, v in data.items()}


def _compile_keyword_regex(ticker: str, keywords: List[str]) -> re.Pattern:
    terms = [ticker, f"${ticker}"] + keywords
    escaped = [re.escape(t) for t in terms if t]
    pattern = r"(" + r"|".join(escaped) + r")"
    return re.compile(pattern, flags=re.IGNORECASE)


def run_reddit_social_pipeline(
    run_cfg: Dict[str, Any],
    *,
    keywords_path: str = "data/interim/keywords.json",
    raw_dir: str = "data/raw/social",
    verbose_print_urls: bool = True,
):
    tickers: List[str] = list(run_cfg["universe"]["tickers"])
    keyword_count: int = int(run_cfg.get("keyword_count", 15))
    max_posts_per_ticker: int = int(run_cfg.get("news_limit_per_ticker", 200))

    cfg = RedditSocialPipelineConfig(
        subreddits=["stocks", "investing", "wallstreetbets", "options"],
        limit=100,
        max_pages=5,
        sleep_s=1.2,
        include_over18=False,
        max_posts_per_ticker=max_posts_per_ticker,
    )

    kw_map = _load_keywords_map(Path(keywords_path))

    raw_dir_p = Path(raw_dir)
    _ensure_dir(raw_dir_p)

    # ONLY output file (stored in RAW directory)
    output_path = raw_dir_p / "reddit_posts.jsonl"

    listing_params = RedditListingParams(
        limit=cfg.limit,
        max_pages=cfg.max_pages,
        sleep_s=cfg.sleep_s,
        include_over18=cfg.include_over18,
        base="https://old.reddit.com",
    )

    # 1) Fetch posts (kept only in memory)
    raw_pool: List[Dict[str, Any]] = []
    scrape_urls: List[str] = []

    for sub in cfg.subreddits:
        posts, urls = fetch_subreddit_new(
            sub,
            params=listing_params,
            return_scrape_urls=True,
        )
        scrape_urls.extend(urls)
        raw_pool.extend(posts)

    raw_pool = dedupe_rows(raw_pool, key="id")

    # 2) Build CLEAN rows
    processed_rows: List[Dict[str, Any]] = []

    for ticker in tickers:
        kws = (kw_map.get(ticker, []) or [])[:keyword_count]
        rx = _compile_keyword_regex(ticker, kws)

        matched_count = 0

        for r in raw_pool:
            text = (r.get("text") or "").strip()
            if not text:
                continue

            m = rx.search(text)
            if not m:
                continue

            source_id = r.get("id")
            row_id = f"reddit:{source_id}:{ticker}"  # unique row id

            processed_rows.append(
                {
                    "id": row_id,
                    "source_id": source_id,
                    "platform": "reddit",
                    "ticker": ticker,
                    "matched_term": m.group(1),
                    "subreddit": r.get("subreddit"),
                    "author": r.get("author"),
                    "created_utc": r.get("created_utc"),
                    "date_utc": r.get("date_utc"),
                    "title": r.get("title"),
                    "selftext": r.get("selftext"),
                    "text": text,
                    "score": r.get("score"),
                    "num_comments": r.get("num_comments"),
                    "permalink": r.get("permalink"),
                    "post_url": r.get("post_url"),
                }
            )

            matched_count += 1
            if matched_count >= cfg.max_posts_per_ticker:
                break

    write_jsonl(processed_rows, str(output_path))

    if verbose_print_urls:
        print("\n=== Reddit scrape URLs ===")
        for u in sorted(set(scrape_urls)):
            print(u)
        print("=== End URLs ===\n")

    print(f"Wrote CLEAN: {output_path} ({len(processed_rows)} rows)")

    return output_path
