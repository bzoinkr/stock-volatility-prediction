from __future__ import annotations

import math
from datetime import datetime, timedelta

from apis.market_data import fetch_price_data


def parkinson_daily_vol_from_row(high, low, annualize=True):
    """
    Parkinson daily volatility (range-based).
    sigma_t = sqrt( (ln(H/L)^2) / (4 ln 2) )
    """
    if high is None or low is None:
        return None

    high = float(high)
    low = float(low)

    if high <= 0 or low <= 0 or high <= low:
        return None

    hl = math.log(high / low)
    vol = math.sqrt((hl ** 2) / (4.0 * math.log(2.0)))

    if annualize:
        vol *= math.sqrt(252)

    return float(vol)


def compute_volatility_for_ticker_range(ticker, start_date, end_date, annualize=True):
    """
    Returns:
        {
            "YYYY-MM-DD": volatility_value,
            ...
        }
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    fetch_start = start_dt - timedelta(days=10)
    fetch_end = end_dt + timedelta(days=1)

    df = fetch_price_data(
        ticker,
        fetch_start.strftime("%Y-%m-%d"),
        fetch_end.strftime("%Y-%m-%d")
    )

    if df is None or df.empty:
        return {}

    results = {}

    for i, dt in enumerate(df.index):
        date_str = dt.strftime("%Y-%m-%d")

        if date_str < start_date or date_str > end_date:
            continue

        try:
            high = df["High"].iloc[i][ticker]
            low = df["Low"].iloc[i][ticker]
            vol = parkinson_daily_vol_from_row(high, low, annualize)
        except Exception:
            vol = None

        results[date_str] = vol

    return results


def compute_volatility_for_tickers(tickers, start_date, end_date, annualize=True):
    results = {}

    for ticker in tickers:
        try:
            ticker_results = compute_volatility_for_ticker_range(
                ticker,
                start_date,
                end_date,
                annualize
            )
            results[ticker] = ticker_results
            print(f"{ticker} -> {len(ticker_results)} dates")
        except Exception as e:
            print(f"{ticker} FAILED: {e}")
            results[ticker] = {}

    return results