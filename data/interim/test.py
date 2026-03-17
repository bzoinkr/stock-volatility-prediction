import yfinance as yf
import json

with open("fortune500_tickers.json", "r") as f:
    data = json.load(f)
tickers = data.get("tickers", [])



 # your list
failed = []
for t in tickers:
    try:
        info = yf.Ticker(t).fast_info
        if not info.get("lastPrice"):
            failed.append(t)
    except:
        failed.append(t)

print("Failed:", failed)