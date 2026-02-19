from apis.ollama_data import get_peer_tickers
from common.paths import INTERIM_DATA_DIR  # Base path for interim (cached) data artifacts
import json, os
from collections import Counter


# --------------------------------------------------
# Build keywords for a list of tickers
# --------------------------------------------------
def build_peerCompanies(
        tickers: list[str],
        k: int = 5,
        runs: int = 30
) -> dict[str, list[str]]:
    result = {}

    print(f"\nStarting peer company generation...")
    print(f"Tickers: {tickers}")
    print(f"Top K peers per ticker: {k}")
    print(f"LLM runs per ticker: {runs}\n")

    for t in tickers:
        print(f"\nGenerating peer companies for ticker: {t}")
        counter = Counter()

        for i in range(runs):
            print(f"  Run {i + 1}/{runs} for {t}...")

            peers = get_peer_tickers(t, k=k)
            print(f"    Returned: {peers}")

            # normalize
            counter.update(set(p.upper().strip() for p in peers))

        top_peers = [p for p, _ in counter.most_common(k)]

        print(f"\nFinal selected peer companies for {t}: {top_peers}")
        result[t] = top_peers

    print("\nPeer company generation complete.\n")

    return result


# --------------------------------------------------
# Save peer ticker dictionary to disk as JSON
# --------------------------------------------------
def save_keywords(
    peer_tickers: dict[str, list[str]],
    filename: str = "peertickers.json"
):
    outpath = os.path.join(INTERIM_DATA_DIR, filename)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(peer_tickers, f, indent=2)

    return outpath
