import requests

headers = {
    'Content-Type': 'application/json'
}

def get_ticker_daily(ticker='NVDA',
    start_date='2026-01-02',
    end_date='2030-01-01',
    resample_freq='daily',
    columns=None,
    token='3d657ef651a029d6e8e71f6670282dfdb8877f8d'):
    """
        Fetches ticker data from Tiingo for specific tickers or tags.
        Returns raw JSON data (list of dicts) from the API.
    """
    if columns is None:
        columns = ['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']

    columns = ",".join(columns)

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    query_params = { 'token' : token, 'columns' : columns, 'startDate' : start_date, 'endDate' : end_date, 'resampleFreq' : resample_freq }
    response = requests.get(url, headers=headers, params=query_params)
    response.raise_for_status()

    data = response.json()

    # Tiingo returns a list of dicts on success, or a single dict on error
    if isinstance(data, dict):
        raise ValueError(f"Tiingo API error for '{ticker}': {data}")
    if not data:
        raise ValueError(f"No data returned for '{ticker}' between {start_date} and {end_date}")

    return data
