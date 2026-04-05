from __future__ import annotations

import datetime
import hashlib
import json
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path

from scripts._bootstrap import *
from pipelines.social.reddit_pipeline_train import run_pipeline
from apis.reddit_data import RateLimitError


# ── Paths ────────────────────────────────────────────────────────────────────
KEYWORDS_PATH = Path("data/interim/keywords.json")
OUTPUT_PATH   = Path("data/raw/social/reddit_posts_train.json")
STATE_PATH    = Path("data/interim/.fetch_state.json")

# ── Config ───────────────────────────────────────────────────────────────────
cfg = {
    "subreddits": ["stocks", "investing", "wallstreetbets", "options"],
    "request_delay_s": 0.1,
    "max_posts_per_ticker": 2000,
}


# ── State helpers ─────────────────────────────────────────────────────────────

def _cfg_hash() -> str:
    """Stable hash of cfg + today's date — invalidates state on a new day or config change."""
    today = datetime.date.today().isoformat()
    serialized = json.dumps({**cfg, "_date": today}, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def load_state() -> dict | None:
    """Return saved state if it exists and cfg hasn't changed, else None."""
    if not STATE_PATH.exists():
        return None
    with open(STATE_PATH) as f:
        state = json.load(f)
    if state.get("cfg_hash") != _cfg_hash():
        print("Config has changed since last run — ignoring saved state, starting fresh.")
        STATE_PATH.unlink()
        return None
    return state


def save_state(ticker: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"resume_from": ticker, "cfg_hash": _cfg_hash()}, f)


def clear_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()


# ── Run ───────────────────────────────────────────────────────────────────────

def _extract_ticker(error_msg: str, remaining_keywords: list) -> str | None:
    """
    Try to identify the failed ticker from the error message by checking
    which tickers in the remaining list appear in the URL/message.
    Returns the first match found, or None.
    """
    for entry in remaining_keywords:
        ticker = entry["ticker"]
        if ticker in error_msg:
            return ticker
    return None


def main() -> None:
    with open(KEYWORDS_PATH) as f:
        all_keywords: list = json.load(f)

    state = load_state()
    if state:
        resume_from = state["resume_from"]
        tickers = [entry["ticker"] for entry in all_keywords]
        if resume_from not in tickers:
            print(f"Saved ticker '{resume_from}' not found in keywords — starting fresh.")
            clear_state()
            random.shuffle(all_keywords)
        else:
            idx = tickers.index(resume_from)
            print(f"Auto-resuming from '{resume_from}' (skipping {idx} ticker(s)).")
            all_keywords = all_keywords[idx:]
    else:
        random.shuffle(all_keywords)

    temp_keywords_path = KEYWORDS_PATH.with_stem(KEYWORDS_PATH.stem + "_resume")
    with open(temp_keywords_path, "w") as f:
        json.dump(all_keywords, f, indent=2)

    try:
        run_pipeline(temp_keywords_path, OUTPUT_PATH, cfg)
    except RateLimitError as e:
        failed_ticker = _extract_ticker(str(e), all_keywords)
        if failed_ticker:
            save_state(failed_ticker)
            print(f"\n429 rate limit hit on '{failed_ticker}'. "
                  f"State saved — re-run the script to resume from this ticker.")
        else:
            print(f"\n429 rate limit hit but could not identify ticker. "
                  f"Re-run the script to retry.")
        sys.exit(1)
    else:
        clear_state()
    finally:
        temp_keywords_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()