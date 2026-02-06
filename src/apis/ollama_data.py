import os, re, time, requests


# ---------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------
# Allows overriding via environment variables if desired
URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "phi4")


# ---------------------------------------------------------
# Generate single-word finance keywords for a stock ticker
# ---------------------------------------------------------
def get_keywords(stock: str, k: int = 15, retries: int = 2) -> list[str]:

    # Prompt instructing the model to output only single-word keywords
    prompt = f"""
        Give {k} single-word keywords for {stock}.
        Output ONLY the words separated by spaces or newlines.
        No numbering, no bullets, no extra text.
        No generic words like stock price market trading investing.
        """.strip()

    # Words we never want returned as keywords
    banned = {"stock", "price", "market", "trading", "investing", "investment", "news", "analysis"}

    # Try multiple times in case the model misbehaves
    for _ in range(retries + 1):

        try:
            # Call Ollama local API
            r = requests.post(
                URL,
                json={"model": MODEL, "stream": False, "prompt": prompt},
                timeout=60
            )
            r.raise_for_status()

            # Raw model output
            text = r.json().get("response", "")

            # Extract single words (letters/numbers only)
            words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]+\b", text.lower())

            out, seen = [], set()

            # Deduplicate, filter banned words, and cap at k keywords
            for w in words:

                if w in banned or w in seen:
                    continue

                seen.add(w)
                out.append(w)

                if len(out) == k:
                    break

            # If we got a reasonable amount, accept result
            if len(out) >= max(3, k // 2):
                return out

        # Any error → wait briefly and retry
        except Exception:
            time.sleep(1)

    # Final fallback if everything fails
    return []


def get_peer_tickers(ticker: str, k: int = 6, retries: int = 2) -> list[str]:

    prompt = f"""
        Give {k} US stock TICKERS of companies operating in the same industry as {ticker}.
        Output ONLY tickers separated by spaces or newlines.
        No punctuation, no bullets, no numbering, no extra text.
        """.strip()

    for _ in range(retries + 1):
        try:
            r = requests.post(URL, json={"model": MODEL, "stream": False, "prompt": prompt}, timeout=60)
            r.raise_for_status()
            text = r.json().get("response", "")

            # Extract uppercase-ish tickers (1-5 chars, allow dot for BRK.B etc.)
            raw = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", text)
            out, seen = [], set()

            for t in raw:
                if t == ticker.upper():
                    continue
                if t in seen:
                    continue
                seen.add(t)
                out.append(t)
                if len(out) == k:
                    break

            if len(out) >= max(2, k // 2):
                return out

        except Exception:
            time.sleep(1)

    return []