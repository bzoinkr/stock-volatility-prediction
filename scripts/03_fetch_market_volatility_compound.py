from _bootstrap import *

import json
from pathlib import Path

from common.config import load_config
from pipelines.market.get_market_volatility_pipeline import compute_volatility_for_tickers





def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    TARGET_DATE = run["universe"]["end_date"]
    ANNUALIZE = True

    INPUT_JSON = Path("data/interim/fortune500_tickers.json")
    OUTPUT_JSON = Path("data/processed/market/volatility.json")

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    tickers = data.get("tickers", [])

    results = compute_volatility_for_tickers(
        tickers,
        TARGET_DATE,
        ANNUALIZE
    )

    output = {
        "date": TARGET_DATE,
        "window": 1,
        "annualized": ANNUALIZE,
        "volatility": results
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
