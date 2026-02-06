from _bootstrap import *

from datetime import date, timedelta

from common.config import load_config
from pipelines.peerCompany_pipeline import build_peerCompanies, save_keywords
from pipelines.yahoo_news_pipeline import fetch_and_save_yahoo_news


def main():
    # Load run configuration (tickers, dates, limits, etc.)
    run = load_config("run.yaml")

    # Base tickers defined in the config universe
    tickers = run["universe"]["tickers"]

    # Fail fast if no tickers were found
    if not tickers:
        raise KeyError(
            f"No tickers found in run.yaml. Available keys: {list(run.keys())}\n"
            "Add one of these keys: tickers / symbols / stocks"
        )

    # --------------------------------------------------
    # 1) Build peer-company universe via Ollama
    # --------------------------------------------------
    peer_k = run.get("peer_count", 5)  # number of peers per ticker
    pC = build_peerCompanies(tickers, k=peer_k)

    # Persist peers for reproducibility/debugging
    path = save_keywords(pC)
    print("Saved:", path)

    # --------------------------------------------------
    # 2) Fetch Yahoo news for the base ticker + peers
    # --------------------------------------------------
    end_date = run["universe"]["end_date"] or date.today().isoformat()
    start_date = run["universe"]["start_date"] or (date.fromisoformat(end_date) - timedelta(days=7)).isoformat()
    limit_per_ticker = int(run.get("news_limit_per_ticker", 200))

    # Fetch and save news (single-ticker peer lookup based on tickers[0])
    path, tickers_used = fetch_and_save_yahoo_news(
        tickers,
        peer_tickers=pC[tickers[0]],
        start_date=start_date,
        end_date=end_date,
        limit_per_ticker=limit_per_ticker,
    )

    print("Saved:", path)
    print("Tickers used:", tickers_used)


if __name__ == "__main__":
    main()
