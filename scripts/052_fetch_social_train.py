from __future__ import annotations

from pathlib import Path

from _bootstrap import *
from pipelines.social.reddit_pipeline_train import run_pipeline



# ── Paths ────────────────────────────────────────────────────────────────────
KEYWORDS_PATH = Path("data/interim/keywords.json")
OUTPUT_PATH = Path("data/raw/social/reddit_posts_train.json")

# ── Config ───────────────────────────────────────────────────────────────────
cfg = {
    "subreddits": ["stocks", "investing", "wallstreetbets", "options"],
    "request_delay_s": 0.1,
    "max_posts_per_ticker": 2000,
}

# ── Run ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run_pipeline(KEYWORDS_PATH, OUTPUT_PATH, cfg)

if __name__ == "__main__":
    main()