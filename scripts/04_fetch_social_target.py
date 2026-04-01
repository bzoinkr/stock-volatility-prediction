from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from _bootstrap import *
from common.config import load_config
from pipelines.social.reddit_pipeline_train import run_pipeline


# ── Paths ────────────────────────────────────────────────────────────────────
KEYWORDS_PATH = Path("data/interim/keywords.json")
OUTPUT_PATH   = Path("data/raw/social/reddit_posts_target.json")

# ── Config ───────────────────────────────────────────────────────────────────
cfg = {
    "subreddits": ["stocks", "investing", "wallstreetbets", "options"],
    "request_delay_s": 0.1,
    "max_posts_per_ticker": 2000,
}

# ── Run ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run = load_config("run.yaml")
    target_tickers = run["universe"]["ticker_symbols"]

    if not target_tickers:
        raise KeyError(
            "No tickers found in run.yaml under 'universe.ticker_symbols'. "
            "Add at least one ticker to this list."
        )

    with open(KEYWORDS_PATH) as f:
        all_keywords: list = json.load(f)

    matched = [entry for entry in all_keywords if entry["ticker"] in target_tickers]
    if not matched:
        raise ValueError(
            f"Ticker(s) {target_tickers} not found in {KEYWORDS_PATH}. "
            "Ensure keywords.json has entries for these tickers."
        )

    missing = [t for t in target_tickers if t not in {e["ticker"] for e in matched}]
    if missing:
        print(f"Warning: no keyword entries found for: {missing} — skipping.")

    print(f"Running Reddit pipeline for: {[e['ticker'] for e in matched]}")

    temp_keywords_path = KEYWORDS_PATH.with_stem(KEYWORDS_PATH.stem + "_subset")
    with open(temp_keywords_path, "w") as f:
        json.dump(matched, f, indent=2)

    try:
        run_pipeline(temp_keywords_path, OUTPUT_PATH, cfg)
    finally:
        temp_keywords_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()