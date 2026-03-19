import os, re, time, requests


# ---------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------
# Allows overriding via environment variables if desired
URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "phi4")


# ---------------------------------------------------------
# Fetch company context from yfinance for prompt grounding
# ---------------------------------------------------------
def _get_company_context(ticker: str) -> dict:
    """
    Pull basic company metadata from yfinance to ground the LLM prompt.
    Falls back to bare ticker if yfinance is unavailable or the lookup fails.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            "name":        info.get("longName", ticker),
            "sector":      info.get("sector", ""),
            "industry":    info.get("industry", ""),
            # Trim description so it doesn't bloat the prompt
            "description": info.get("longBusinessSummary", "")[:500],
        }
    except Exception:
        return {"name": ticker, "sector": "", "industry": "", "description": ""}


# ---------------------------------------------------------
# Generate single-word finance keywords for a stock ticker
# ---------------------------------------------------------
def get_keywords(stock: str, k: int = 15, retries: int = 2) -> list[str]:

    # Ground the prompt in real company data so the model isn't guessing
    ctx = _get_company_context(stock)

    # Build a context block only from fields that are actually populated
    context_lines = [f"{ctx['name']} ({stock})"]
    if ctx["sector"]:
        context_lines.append(f"Sector: {ctx['sector']}  |  Industry: {ctx['industry']}")
    if ctx["description"]:
        context_lines.append(f"Business: {ctx['description']}")
    context_block = "\n".join(context_lines)

    prompt = f"""
# Role
Expert financial research analyst specialising in news-driven signals.

# Company
{context_block}

# Task
Generate {k} high-signal search keywords that would surface news SPECIFICALLY about {ctx['name']}.

# Rules
- Every keyword must be something that could plausibly appear in a headline uniquely about {ctx['name']}
- Prefer named entities: specific product lines, executive names, key suppliers or partners,
  regulatory bodies with jurisdiction over this company, named lawsuits or investigations,
  and material geographic markets
- AVOID generic financial terms: earnings, revenue, outlook, growth, guidance, results
- AVOID sector-wide terms that apply to any company in this industry
- AVOID the company name and ticker itself as a keyword

# Output
Return ONLY the keywords separated by spaces. No numbering, no bullets, no explanation.
""".strip()

    # Derive a single-word version of the company name to prepend as a keyword.
    # Take the first word and strip any non-alphanumeric characters (e.g. "Apple" from "Apple Inc.")
    company_keyword = re.sub(r"[^a-zA-Z0-9]", "", ctx["name"].split()[0]).lower() if ctx["name"] != stock else ""

    # Words we never want returned as keywords
    banned = {
        "stock", "price", "market", "trading", "investing", "investment",
        "news", "analysis", "supply", "chain", "earnings", "revenue",
        "outlook", "guidance", "growth", "results", "shares", "quarterly",
        stock.lower(), ctx["name"].lower(),
    }

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

            # Extract single words (letters/numbers only, min length 3)
            words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]+\b", text.lower())

            # Seed output with the company name keyword so it's always first
            out  = [company_keyword] if company_keyword else []
            seen = set(out)

            # Deduplicate, filter banned words, and cap at k keywords
            for w in words:

                # Skip anything in banned set or a substring of the company name
                if w in banned or w in seen or len(w) < 3:
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

    # Use company name if available for a clearer prompt
    ctx = _get_company_context(ticker)
    company_label = f"{ctx['name']} ({ticker})" if ctx["name"] != ticker else ticker

    prompt = f"""
Give {k} US stock TICKERS of companies operating in the same industry as {company_label}.
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