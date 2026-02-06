from apis.ollama_data import get_peer_tickers
from common.paths import NEWS_RAW_DATA_DIR  # Base path for interim (cached) data artifacts
import json, os


# --------------------------------------------------
# Build keywords for a list of tickers
# --------------------------------------------------
def build_peerCompanies(tickers: list[str], k: int = 5) -> dict[str, list[str]]:
    return {t: get_peer_tickers(t, k=k) for t in tickers}


# --------------------------------------------------
# Save peer ticker dictionary to disk as JSON
# --------------------------------------------------
def save_keywords(
    peer_tickers: dict[str, list[str]],
    filename: str = "peertickers.json"
):
    outpath = os.path.join(NEWS_RAW_DATA_DIR, filename)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(peer_tickers, f, indent=2)

    return outpath
