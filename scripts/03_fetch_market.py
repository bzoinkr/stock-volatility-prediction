from _bootstrap import *

from common.config import load_config
from pipelines.market_pipeline import run_market_pipeline


def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    ticker_symbols = run["universe"]["ticker_symbols"]
    start_date = run["universe"]["start_date"]
    end_date = run["universe"]["end_date"]

    result = run_market_pipeline(
        ticker_symbol=ticker_symbols[0],
        start_date=start_date,
        end_date=end_date,
    )

    print(f"Stock data written to: {result['stock_output']} ({result['stock_rows']} rows)")
    print(f"VIX data written to:   {result['vix_output']} ({result['vix_rows']} rows)")


if __name__ == "__main__":
    main()
