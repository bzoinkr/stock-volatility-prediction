from _bootstrap import *

from datetime import date, timedelta

from common.config import load_config
from pipelines.peerCompany_pipeline import build_peerCompanies, save_keywords
from pipelines.finnhub_news_pipeline import fetch_and_save_finnhub_news


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
    llm_runs = run.get("llm_run_count", 30)   # Number of times to run the llm
    pC = build_peerCompanies(tickers, k=peer_k, runs=llm_runs)

    # Persist peers for reproducibility/debugging
    path = save_keywords(pC)
    print("Saved:", path)

    # --------------------------------------------------
    # 2) Fetch news for the base ticker + peers (saved to yahoo_news.jsonl)
    # --------------------------------------------------
    end_date = run["universe"]["end_date"] or date.today().isoformat()
    start_date = run["universe"]["start_date"] or (date.fromisoformat(end_date) - timedelta(days=7)).isoformat()
    limit_per_ticker = int(run.get("news_limit_per_ticker", 200))
    peer_list = pC[tickers[0]]

    path, tickers_used = fetch_and_save_finnhub_news(
        tickers,
        peer_tickers=peer_list,
        start_date=start_date,
        end_date=end_date,
        limit_per_ticker=limit_per_ticker,
    )
    print("Saved:", path)
    print("Tickers used:", tickers_used)


if __name__ == "__main__":
    main()
