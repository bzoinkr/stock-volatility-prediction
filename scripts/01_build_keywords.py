from idlelib.iomenu import encoding

from _bootstrap import *

from common.config import load_config
from pipelines.keywords_pipeline import build_keywords, save_keywords

import json
import sys

def main(train=False):
    # Load run configuration
    run = load_config("run.yaml")

    # Fetch main ticker
    if train == False:
        tickers = run["universe"]["ticker_symbols"]
        print(tickers)
    else:
        ticker_path = "data/interim/fortune500_tickers.json"
        with open(ticker_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tickers = data["tickers"]
        print(tickers)

    # Fail fast if no tickers were found
    if not tickers:
        raise KeyError(
            f"No tickers found in run.yaml. Available keys: {list(run.keys())}\n"
            "Add one of these keys: tickers / symbols / stocks"
        )

    KEYWORDS_PATH = Path("data/interim/keywords.json")
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    loadedTickers = [_["ticker"] for _ in data]

    newTickers = []
    for t in tickers:
        if t not in loadedTickers:
            newTickers.append(t)


    # Number of keywords per ticker (default = 15)
    k = run.get("keyword_count", 5)
    # Number of times to run the llm
    llm_runs = run.get("llm_run_count", 1)

    # Build and save keywords
    kw = build_keywords(newTickers, k=k, runs=llm_runs)
    path = save_keywords(kw)

    print("Saved:", path)
    print("If you don't see the ticker you mentioned in the above list, its keywords have already been generated")


if __name__ == "__main__":
    main(train="--train" in sys.argv)
