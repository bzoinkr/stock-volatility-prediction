from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import *
from pipelines.social.reddit_pipeline_train import run_pipeline


# ── Paths ────────────────────────────────────────────────────────────────────
KEYWORDS_PATH = Path("data/interim/keywords.json")
OUTPUT_PATH   = Path("data/raw/social/reddit_posts_train.json")

# ── Config ───────────────────────────────────────────────────────────────────
cfg = {
    "subreddits": ["stocks", "investing", "wallstreetbets", "options"],
    "request_delay_s": 0.1,
    "max_posts_per_ticker": 2000,
}

# ── Run ───────────────────────────────────────────────────────────────────────

def main(resume_from: str | False = False) -> None:
    with open(KEYWORDS_PATH) as f:
        all_keywords: list = json.load(f)

    if resume_from:
        tickers = [entry["ticker"] for entry in all_keywords]
        if resume_from not in tickers:
            raise ValueError(f"Ticker '{resume_from}' not found in {KEYWORDS_PATH}")
        idx = tickers.index(resume_from)
        print(f"Resuming from '{resume_from}' (row {idx}) — skipping {idx} ticker(s): {tickers[:idx]}")
        all_keywords = all_keywords[idx:]

    # Write to a temp path so the original is never mutated
    temp_keywords_path = KEYWORDS_PATH.with_stem(KEYWORDS_PATH.stem + "_resume")
    with open(temp_keywords_path, "w") as f:
        json.dump(all_keywords, f, indent=2)

    try:
        run_pipeline(temp_keywords_path, OUTPUT_PATH, cfg)
    finally:
        temp_keywords_path.unlink(missing_ok=True)

if __name__ == "__main__":
    if "--resume_from" in sys.argv:
        idx = sys.argv.index("--resume_from")
        main(resume_from=sys.argv[idx + 1])
    else:
        main()