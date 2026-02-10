import requests
import pandas as pd

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
    """
    if columns is None:
        columns = ['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']

    columns = ",".join(columns)

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    query_params = { 'token' : token, 'columns' : columns, 'startDate' : start_date, 'endDate' : end_date, 'resampleFreq' : resample_freq }
    response = requests.get(url, headers=headers, params=query_params)

    df = pd.DataFrame(response.json())
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df
