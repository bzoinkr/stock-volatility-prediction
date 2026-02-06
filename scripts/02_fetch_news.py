from _bootstrap import *

from common.config import load_config
from pipelines.peerCompany_pipeline import build_peerCompanies, save_keywords

import json
import os
from datetime import date, timedelta


def main():
    # Load run configuration
    run = load_config("run.yaml")

    # Fetch main ticker
    tickers = run["universe"]["tickers"]

    # Fail fast if no tickers were found
    if not tickers:
        raise KeyError(
            f"No tickers found in run.yaml. Available keys: {list(run.keys())}\n"
            "Add one of these keys: tickers / symbols / stocks"
        )


    # Number of peers per ticker (default = 5)
    k = run.get("Peer Company Count", 5)

    # Build and save keywords
    pC = build_peerCompanies(tickers, k=k)
    path = save_keywords(pC)

    print("Saved:", path)

    print(pC)


if __name__ == "__main__":
    main()
