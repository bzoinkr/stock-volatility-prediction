from _bootstrap import *

from common.config import load_config
from pipelines.keywords_pipeline import build_keywords, save_keywords


def main():
    # Load run configuration
    run = load_config("run.yaml")

    # Try common key names for tickers
    tickers = run["universe"]["tickers"]

    # Fail fast if no tickers were found
    if not tickers:
        raise KeyError(
            f"No tickers found in run.yaml. Available keys: {list(run.keys())}\n"
            "Add one of these keys: tickers / symbols / stocks"
        )

    # Number of keywords per ticker (default = 15)
    k = run.get("keyword_count", 15)

    # Build and save keywords
    kw = build_keywords(tickers, k=k)
    path = save_keywords(kw)

    print("Saved:", path)


if __name__ == "__main__":
    main()
