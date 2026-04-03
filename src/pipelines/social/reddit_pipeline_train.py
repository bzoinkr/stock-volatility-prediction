from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Set

import requests

from apis.reddit_data import fetch_posts_for_ticker, RateLimitError


def _load_keywords_map(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(item["ticker"]).strip(): [str(k) for k in item.get("keywords", []) if k]
        for item in data if item.get("ticker")
    }


def _read_existing_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rid = json.loads(line).get("id")
                if rid:
                    ids.add(rid)
            except Exception:
                continue
    return ids


def _append_jsonl(rows, path: Path) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_pipeline(keywords_path: Path, output_path: Path, cfg: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kw_map = _load_keywords_map(keywords_path)
    existing_ids = _read_existing_ids(output_path)

    session = requests.Session()
    session.headers.update({"User-Agent": "stock-volatility-pipeline/1.0"})

    appended_total = 0
    tickers = list(kw_map.keys())

    for idx, ticker in enumerate(tickers, 1):
        try:
            posts = fetch_posts_for_ticker(
                ticker,
                kw_map[ticker],
                subreddits=cfg["subreddits"],
                max_posts=cfg["max_posts_per_ticker"],
                session=session,
            )
        except RateLimitError:
            raise  # propagate immediately — do not swallow
        except Exception as e:
            print(f"[{idx}/{len(tickers)}] {ticker} failed: {e}")
            time.sleep(cfg["request_delay_s"])
            continue

        rows_to_append = []
        seen = set()
        for post in posts:
            row_id = f"reddit:{post['id']}:{ticker}"
            if row_id in existing_ids or post["id"] in seen:
                continue
            seen.add(post["id"])
            existing_ids.add(row_id)
            rows_to_append.append({**post, "id": row_id, "source_id": post["id"]})

        _append_jsonl(rows_to_append, output_path)
        appended_total += len(rows_to_append)
        print(f"[{idx}/{len(tickers)}] {ticker} -> {len(rows_to_append)} appended")
        time.sleep(cfg["request_delay_s"])

    print(f"\nDone. {appended_total} total rows appended.")