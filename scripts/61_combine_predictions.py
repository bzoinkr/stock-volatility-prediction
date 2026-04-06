from _bootstrap import *

import json
from pathlib import Path
from datetime import datetime

from common.config import load_config


# ── Paths ─────────────────────────────────────────────────────────────────────

NEW_PATH = "artifacts/predictions/news_prediction.json"
SOCIAL_PATH = "artifacts/predictions/social_prediction.json"
OUTPUT_PATH = "artifacts/predictions/final_prediction.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)


def get_latest_prediction(entries: list) -> dict | None:
    """Return the entry with the highest run number for a given ticker/date list."""
    if not entries:
        return None
    return max(entries, key=lambda e: e["run"])


def get_next_run(existing_output: dict, ticker: str, target_date: str) -> int:
    """Return the next run number for a ticker/date in the existing output file."""
    existing_entries = existing_output.get(ticker, {}).get(target_date, [])
    if not existing_entries:
        return 1
    return max(e["run"] for e in existing_entries) + 1


def save_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run = load_config("run.yaml")

    tickers = run["universe"]["ticker_symbols"]
    prediction_ratio = run["prediction_ratio"]  # 0.0 = fully social, 1.0 = fully news
    target_date = run["universe"]["target_date"]

    news_data = load_json(NEW_PATH)
    social_data = load_json(SOCIAL_PATH)
    existing_output = load_json(OUTPUT_PATH)

    output = existing_output
    skipped = []
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    newly_stored = {}

    for ticker in tickers:
        news_entries = news_data.get(ticker, {}).get(target_date, [])
        social_entries = social_data.get(ticker, {}).get(target_date, [])

        missing = []
        if not news_entries:
            missing.append("news")
        if not social_entries:
            missing.append("social")

        if missing:
            skipped.append((ticker, missing))
            continue

        news_pred = get_latest_prediction(news_entries)
        social_pred = get_latest_prediction(social_entries)

        news_val = news_pred["predicted"]
        social_val = social_pred["predicted"]
        blended_val = (prediction_ratio * news_val) + ((1 - prediction_ratio) * social_val)

        next_run = get_next_run(existing_output, ticker, target_date)

        entry = {
            "ticker": ticker,
            "date": target_date,
            "predicted": blended_val,
            "news_predicted": news_val,
            "social_predicted": social_val,
            "prediction_ratio": prediction_ratio,
            "run": next_run,
            "timestamp": timestamp_now,
        }

        output.setdefault(ticker, {}).setdefault(target_date, []).append(entry)
        newly_stored.setdefault(ticker, {}).setdefault(target_date, []).append(entry)

    # ── Print results ─────────────────────────────────────────────────────────

    if skipped:
        print("\n── Skipped Tickers ──────────────────────────────────────────")
        for ticker, missing_sources in skipped:
            print(f"  {ticker}: no data found in [{', '.join(missing_sources)}] for {target_date}")

    if newly_stored:
        save_json(OUTPUT_PATH, output)
        print("\n── Final Predictions Stored ─────────────────────────────────")
        print(f"  Location : {OUTPUT_PATH}")
        print(f"  Date     : {target_date}")
        print(f"  Ratio    : {prediction_ratio} (0=fully social, 1=fully news)")
        print(f"  Tickers  : {len(newly_stored)} stored\n")
        for ticker, dates in newly_stored.items():
            for date, entries in dates.items():
                e = entries[0]
                print(f"  {ticker}  |  run={e['run']}  blended={e['predicted']:.6f}  "
                      f"news={e['news_predicted']:.6f}  "
                      f"social={e['social_predicted']:.6f}")
        print()
    else:
        print(f"\n  No predictions stored — no tickers had data in both sources for {target_date}.\n")


if __name__ == "__main__":
    main()