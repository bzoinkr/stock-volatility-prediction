import yfinance as yf


def fetch_price_data(ticker, start_date, end_date):
    """
    Fetch OHLC data for a ticker between start_date and end_date.
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    if df is None or df.empty:
        return None

    return df.sort_index()