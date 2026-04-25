import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts._bootstrap import *

import json
from pathlib import Path

from common.config import load_config
from pipelines.market.get_market_volatility_pipeline import compute_volatility_for_tickers


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


def main():
    run = load_config("run.yaml")

    start_date = run["universe"]["start_date_news"]
    end_date = run["universe"]["end_date"]
    annualize = True
    tickers = _selected_tickers(run)

    output_json = Path("data/processed/market/volatilitySingleTicker.json")

    results = compute_volatility_for_tickers(
        tickers,
        start_date,
        end_date,
        annualize,
    )

    if output_json.exists():
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    for ticker, date_map in results.items():
        existing_data.setdefault(ticker, {})
        for date_str, vol in date_map.items():
            existing_data[ticker][date_str] = vol

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Computed volatility for {len(tickers)} tickers.")
    print(f"\nSaved to {output_json.resolve()}")


if __name__ == "__main__":
    main()
