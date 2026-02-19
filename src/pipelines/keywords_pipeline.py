from apis.ollama_data import get_keywords
from common.paths import INTERIM_DATA_DIR  # Base path for interim (cached) data artifacts
import json, os
from collections import Counter


# --------------------------------------------------
# Build keywords for a list of tickers
# --------------------------------------------------
from collections import Counter


def build_keywords(
        tickers: list[str],
        k: int = 15,
        runs: int = 30
) -> dict[str, list[str]]:
    result = {}

    print(f"\nStarting keyword generation...")
    print(f"Tickers: {tickers}")
    print(f"Top K per ticker: {k}")
    print(f"LLM runs per ticker: {runs}\n")

    for t in tickers:
        print(f"\nGenerating keywords for ticker: {t}")
        counter = Counter()

        for i in range(runs):
            print(f"  Run {i + 1}/{runs} for {t}...")

            keywords = get_keywords(t, k=k)
            print(f"    Returned: {keywords}")

            # normalize slightly
            counter.update(set(kw.lower().strip() for kw in keywords))

        top_keywords = [kw for kw, _ in counter.most_common(k)]

        print(f"\nFinal selected keywords for {t}: {top_keywords}")
        result[t] = top_keywords

    print("\nKeyword generation complete.\n")

    return result


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
