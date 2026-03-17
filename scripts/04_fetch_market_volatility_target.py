from _bootstrap import *


from common.config import load_config
from pipelines.market.get_market_volatility_pipeline import compute_volatility_for_tickers



def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    tickers = run["universe"]["ticker_symbols"]
    TARGET_DATE = run["universe"]["end_date"]
    ANNUALIZE = True

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

    print("Target Volatility: ", output["volatility"])


if __name__ == "__main__":
    main()
