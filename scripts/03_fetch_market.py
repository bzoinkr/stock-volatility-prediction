from _bootstrap import *

from datetime import date, timedelta

from common.config import load_config
from apis.market_data import get_ticker_daily
from apis.vix_data import get_vix_data


def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    # Base tickers defined in the config universe
    tickers = run["universe"]["tickers"]
    start_date = run["universe"]["start_date"]
    end_date = run["universe"]["end_date"]
    interval = run["frequency"]["base"] # might need to be 1d instead of 1D

    stock_df = get_ticker_daily(ticker=tickers,
                  start_date=start_date,
                  end_date=end_date,
                  resample_freq='daily',
                  columns=None,
                  token='3d657ef651a029d6e8e71f6670282dfdb8877f8d')
  
    vix_df = get_vix_data(period="1mo", interval=interval)

    print(stock_df.head)
  
    print(vix_df.head)
if __name__ == "__main__":
    main()

