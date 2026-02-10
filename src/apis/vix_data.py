import yfinance as yf

def get_vix_data(period="1mo", interval="1d"):
    """
    Fetches historical VIX data from Yahoo Finance.
    
    Parameters:
    period (str): Data period to download (e.g., '1d', '5d', '1mo', '1y', 'max')
    interval (str): Data interval (e.g., '1m', '2m', '5m', '15m', '1h', '1d')
    
    Returns:
    pandas.DataFrame: Historical VIX prices
    """
    # The VIX ticker on Yahoo Finance is ^VIX
    vix = yf.Ticker("^VIX")
    
    # Fetch historical data
    df = vix.history(period=period, interval=interval)
    
    return df
