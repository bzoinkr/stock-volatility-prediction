from _bootstrap import *

import json
from pathlib import Path


from common.config import load_config
from pipelines.market.get_market_volatility_pipeline import compute_volatility_for_tickers



def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    START_DATE = run["universe"]["start_date_news"]
    END_DATE = run["universe"]["end_date"]
    ANNUALIZE = True

    OUTPUT_JSON = Path("data/processed/market/volatilitySingleTicker.json")

    tickers = run["universe"]["ticker_symbols"]

    results = compute_volatility_for_tickers(
        tickers,
        START_DATE,
        END_DATE,
        ANNUALIZE
    )

    # Read existing output file if it exists
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Store as ticker -> date -> volatility
    for ticker, date_map in results.items():
        if ticker not in existing_data:
            existing_data[ticker] = {}

        for date_str, vol in date_map.items():
            existing_data[ticker][date_str] = vol

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)

    print(f"\nSaved to {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
