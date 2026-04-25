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
    """
    Return saved state if it exists and cfg/date hash matches, else None.
    State schema:
        {
            "cfg_hash": str,
            "completed": [ticker, ...]   # all tickers fully processed this session
        }
    """
    if not STATE_PATH.exists():
        return None
    with open(STATE_PATH) as f:
        state = json.load(f)
    if state.get("cfg_hash") != _cfg_hash():
        print("Config or date has changed since last run — ignoring saved state, starting fresh.")
        STATE_PATH.unlink()
        return None
    return state


def _write_state(completed: set) -> None:
    """Persist the current completed-ticker set to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"cfg_hash": _cfg_hash(), "completed": sorted(completed)}, f, indent=2)


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

    # ── Determine which tickers are already done ──────────────────────────────
    state = load_state()
    completed: set = set(state.get("completed", [])) if state else set()

    if completed:
        print(f"Resuming — {len(completed)} ticker(s) already completed, skipping them.")
    else:
        # Fresh run: shuffle for variety across sessions
        random.shuffle(all_keywords)

    # Filter down to only what still needs fetching
    pending_keywords = [e for e in all_keywords if e["ticker"] not in completed]
    total_all = len(all_keywords)
    total_pending = len(pending_keywords)

    if not pending_keywords:
        print("All tickers already completed. Nothing to do.")
        clear_state()
        return

    print(f"{total_pending} ticker(s) remaining out of {total_all}.")

    # ── Write a temp keywords file for the pipeline ───────────────────────────
    temp_keywords_path = KEYWORDS_PATH.with_stem(KEYWORDS_PATH.stem + "_resume")
    with open(temp_keywords_path, "w") as f:
        json.dump(pending_keywords, f, indent=2)

    # ── Run pipeline, updating completed set after each ticker ────────────────
    try:
        run_pipeline(
            temp_keywords_path,
            OUTPUT_PATH,
            cfg,
            on_ticker_done=lambda ticker: _write_state(completed | {ticker}),
            completed=completed,
        )
    except RateLimitError as e:
        failed_ticker = _extract_ticker(str(e), pending_keywords)
        # completed set was already flushed to disk after every successful ticker,
        # so just report and exit — nothing extra to save.
        if failed_ticker:
            print(f"\n429 rate limit hit on '{failed_ticker}'. "
                  f"Re-run the script to resume — {len(completed)} ticker(s) already saved.")
        else:
            print(f"\n429 rate limit hit (ticker unidentified). "
                  f"Re-run the script to resume — {len(completed)} ticker(s) already saved.")
        sys.exit(1)
    else:
        print(f"\nAll tickers finished.")
        clear_state()
    finally:
        temp_keywords_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()