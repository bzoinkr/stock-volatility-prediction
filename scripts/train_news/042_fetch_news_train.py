import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts._bootstrap import *

import json
import subprocess
import time
from pathlib import Path

from common.config import load_config


def _selected_tickers(run: dict) -> list[str]:
    input_json = Path("data/interim/fortune500_tickers.json")
    with open(input_json, "r", encoding="utf-8") as f:
        raw_tickers = [str(t).upper() for t in json.load(f).get("tickers", []) if t]

    # De-dupe while preserving file order.
    seen = set()
    tickers = []
    for ticker in raw_tickers:
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    # Keep target ticker in the universe even if it's not in the source file.
    target_ticker = run["universe"]["ticker_symbols"][0]
    if target_ticker not in seen:
        tickers.append(target_ticker)
    return tickers


def main() -> None:
    run = load_config("run.yaml")
    tickers = _selected_tickers(run)

    output_path = Path("data/raw/news/yahoo_news_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    script_path = Path("scripts/02_fetch_news.py")
    pause_seconds = 20

    for idx, ticker in enumerate(tickers):
        env = os.environ.copy()
        env["NEWS_TICKERS"] = ticker
        env["NEWS_OUTPUT_FILENAME"] = output_path.name
        env["NEWS_APPEND"] = "1" if idx > 0 else "0"

        print(f"[{idx + 1}/{len(tickers)}] Fetching news for {ticker}...")
        subprocess.run([sys.executable, str(script_path)], check=True, env=env)

        if idx < len(tickers) - 1:
            print(f"Pausing {pause_seconds}s before next ticker...")
            time.sleep(pause_seconds)

    print(f"Fetched news for {len(tickers)} tickers.")
    print("Saved:", output_path.resolve())


if __name__ == "__main__":
    main()
