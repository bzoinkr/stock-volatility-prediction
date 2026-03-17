from __future__ import annotations

import math
from datetime import datetime, timedelta

from apis.market_data import fetch_price_data


def parkinson_daily_vol(ticker, target_date, annualize=True):
    """
    Parkinson daily volatility (range-based).
    sigma_t = sqrt( (ln(H/L)^2) / (4 ln 2) )
    """
    end_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=10)

    df = fetch_price_data(
        ticker,
        start_dt.strftime("%Y-%m-%d"),
        (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    )

    if df is None or df.empty:
        return None

    dates = df.index.strftime("%Y-%m-%d").tolist()

    if target_date not in dates:
        return None

    i = dates.index(target_date)

    try:
        high = float(df["High"].iloc[i][ticker])
        low = float(df["Low"].iloc[i][ticker])
    except Exception:
        return None

    if high <= 0 or low <= 0 or high <= low:
        return None

    hl = math.log(high / low)
    vol = math.sqrt((hl ** 2) / (4.0 * math.log(2.0)))

    if annualize:
        vol *= math.sqrt(252)

    return float(vol)


def compute_volatility_for_tickers(tickers, target_date, annualize=True):
    results = {}

    for ticker in tickers:
        try:
            vol = parkinson_daily_vol(ticker, target_date, annualize)
            results[ticker] = vol
            print(f"{ticker} -> {vol}")
        except Exception as e:
            print(f"{ticker} FAILED: {e}")
            results[ticker] = None

    return results