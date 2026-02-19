from _bootstrap import *

from datetime import date, timedelta

from common.config import load_config
from pipelines.peerCompany_pipeline import build_peerCompanies, save_keywords
from pipelines.yahoo_news_pipeline import fetch_and_save_yahoo_news


def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    # Base tickers defined in the config universe
    tickers = run["universe"]["tickers"]

    stock_df = get_ticker_daily(ticker='NVDA',
                  start_date='2026-01-02',
                  end_date='2030-01-01',
                  resample_freq='daily',
                  columns=None,
                  token='3d657ef651a029d6e8e71f6670282dfdb8877f8d')
  
    vix_df = get_vix_data(period="1mo", interval="1d")

    print(stock_df.head)
  
    print(vix_df.head)
if __name__ == "__main__":
    main()
