from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from apis.market_data import get_ticker_daily
from apis.vix_data import get_vix_data


OUTPUT_DIR = Path("data/processed/market")
STOCK_OUTPUT_PATH = OUTPUT_DIR / "stock_data.xlsx"
VIX_OUTPUT_PATH = OUTPUT_DIR / "vix_data.xlsx"


def run_market_pipeline(
    ticker_symbol: str,
    start_date: str,
    end_date: str,
    token: str = "3d657ef651a029d6e8e71f6670282dfdb8877f8d",
) -> Dict[str, Any]:
    """
    Fetches stock and VIX data, then writes each to a separate Excel file.

    Returns a summary dict with output paths and row counts.
    """
    # Fetch raw data from APIs
    stock_data = get_ticker_daily(
        ticker=ticker_symbol,
        start_date=start_date,
        end_date=end_date,
        resample_freq="daily",
        columns=None,
        token=token,
    )

    vix_data = get_vix_data(start_date=start_date, end_date=end_date)

    # Convert to DataFrames for Excel output
    stock_df = pd.DataFrame(stock_data)
    vix_df = pd.DataFrame(vix_data)

    # Strip timezone info from datetime columns (Excel doesn't support tz-aware datetimes)
    for df in [stock_df, vix_df]:
        for col in df.select_dtypes(include=["datetimetz"]).columns:
            df[col] = df[col].dt.tz_localize(None)

    # Write to separate Excel files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_df.to_excel(STOCK_OUTPUT_PATH, index=False, engine="openpyxl")
    vix_df.to_excel(VIX_OUTPUT_PATH, index=False, engine="openpyxl")

    return {
        "stock_output": str(STOCK_OUTPUT_PATH),
        "vix_output": str(VIX_OUTPUT_PATH),
        "stock_rows": len(stock_df),
        "vix_rows": len(vix_df),
    }
