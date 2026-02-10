from apis.ollama_data import get_keywords
from common.paths import INTERIM_DATA_DIR  # Base path for interim (cached) data artifacts
import json, os


# --------------------------------------------------
# Build keywords for a list of tickers
# --------------------------------------------------
def build_keywords(tickers: list[str], k: int = 15) -> dict[str, list[str]]:
    return {t: get_keywords(t, k=k) for t in tickers}


# --------------------------------------------------
# Save keyword dictionary to disk as JSON
# --------------------------------------------------
def save_keywords(
    keywords_by_ticker: dict[str, list[str]],
    filename: str = "keywords.json"
):
    outpath = os.path.join(INTERIM_DATA_DIR, filename)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(keywords_by_ticker, f, indent=2)

    return outpath
