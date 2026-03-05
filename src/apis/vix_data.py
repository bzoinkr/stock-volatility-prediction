import yfinance as yf

def get_vix_data(start_date, end_date):
    """
    Fetches historical VIX data from Yahoo Finance.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    list[dict]: Raw historical VIX price records
    """
    # The VIX ticker on Yahoo Finance is ^VIX
    df = yf.download("^VIX", start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No VIX data returned between {start_date} and {end_date}")

    # Flatten multi-level columns that yf.download returns
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # Convert Timestamp to ISO string for raw output
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    return df.to_dict(orient='records')
